from utils import config

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import importlib
import random
import time
import os

import torch
import numpy as np
import gradio as gr

from utils.dual_diffusion_utils import init_cuda, save_audio, load_audio, dict_str, get_available_torch_devices
from modules.module import DualDiffusionModule
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline


@dataclass
class GradioAppConfig:
    model_name: str
    model_load_options: dict
    gpu_concurrency_limit: int = 1
    gpu_max_queue_size: int = 10
    
    web_server_host: Optional[str] = None
    web_server_port: int = 3001
    web_server_share: bool = False
    web_server_default_concurrency_limit: int = 1
    web_server_max_queue_size: int = 10

    enable_debug_logging: bool = False

class GradioApp:

    def __init__(self) -> None:

        self.config = GradioAppConfig(**config.load_json(
            os.path.join(config.CONFIG_PATH, "sampling", "gradio_app.json")))
        
        self.init_logging()
        self.logger.debug(f"GradioAppConfig:\n{dict_str(self.config.__dict__)}")

        # load model
        model_path = os.path.join(config.MODELS_PATH, self.config.model_name)
        model_load_options = self.config.model_load_options

        self.logger.info(f"Loading DualDiffusion model from '{model_path}'...")
        pipeline = DualDiffusionPipeline.from_pretrained(model_path, **model_load_options)

        # remove games with no samples in training set
        for game_name, count in pipeline.dataset_info["game_train_sample_counts"].items():
            if count == 0: pipeline.dataset_game_ids.pop(game_name)

        self.pipeline = pipeline

    def init_logging(self) -> None:

        self.logger = logging.getLogger(name="gradio_app")

        if self.config.enable_debug_logging:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        if config.DEBUG_PATH is not None:
            logging_dir = os.path.join(config.DEBUG_PATH, "gradio_app")
            os.makedirs(logging_dir, exist_ok=True)

            datetime_str = datetime.now().strftime(r"%Y-%m-%d_%H_%M_%S")
            self.log_path = os.path.join(logging_dir, f"gradio_app_{datetime_str}.log")
            
            logging.basicConfig(
                handlers=[
                    logging.FileHandler(self.log_path),
                    logging.StreamHandler()
                ],
                format="",
            )
            self.logger.info(f"\nStarted gradio_app at {datetime_str}")
            self.logger.info(f"Logging to {self.log_path}")
        else:
            self.log_path = None
            self.logger.warning("WARNING: DEBUG_PATH not defined, logging disabled")

    def run(self) -> None:

        with gr.Blocks() as self.settings_interface:

            with gr.Row():
                
                model_list = [model_name for model_name in os.listdir(config.MODELS_PATH)
                              if os.path.isdir(os.path.join(config.MODELS_PATH, model_name))]
                model_dropdown = gr.Dropdown(
                    choices=model_list,
                    label="Select a model",
                    value=self.config.model_name,
                    interactive=True,
                    scale=3
                )

                #load_latest_checkpoints_checkbox = gr.Checkbox(label="Load Latest Checkpoints", value=self.config.load_latest_checkpoints, interactive=True)
                #fp16_checkbox = gr.Checkbox(label="fp16", value=self.config.fp16, interactive=True)

                device_dropdown = gr.Dropdown(
                    choices=get_available_torch_devices(),
                    label="Device",
                    value=self.config.model_load_options["device"],
                    interactive=True,
                    scale=1
                )

            with gr.Row():
                
                module_classes: dict[str, type[DualDiffusionModule]] = {}
                model_index = config.load_json(os.path.join(config.MODELS_PATH, model_dropdown.value, "model_index.json"))
                for module_name, module_import_dict in model_index["modules"].items():
                    module_package = importlib.import_module(module_import_dict["package"])
                    module_classes[module_name] = getattr(module_package, module_import_dict["class"])

                model_inventory = DualDiffusionPipeline.get_model_inventory(os.path.join(config.MODELS_PATH, model_dropdown.value))
                for module_name, module_inventory in model_inventory.items():
                    with gr.Column():

                        if module_classes[module_name].has_trainable_parameters:
                            checkpoint_dropdown = gr.Dropdown(
                                choices=["None"] + module_inventory.checkpoints,
                                label=f"Select a {module_name} Checkpoint",
                                value=module_inventory.checkpoints[-1] if len(module_inventory.checkpoints) > 0 else "None",
                                interactive=True,
                            )

                            current_checkpoint = "" if checkpoint_dropdown.value == "None" else checkpoint_dropdown.value
                            ema_dropdown = gr.Dropdown(
                                choices=["None"] + module_inventory.emas[current_checkpoint],
                                label="Select an EMA",
                                value="None",
                                interactive=True,
                            )

        with gr.Blocks() as self.generation_interface:

            # ********** parameter editor **********

            with gr.Row() as self.parameter_editor:
                
                # general params

                with gr.Column(min_width=100):
                    seed = gr.Number(label="Seed", value=42, minimum=0, maximum=99900, precision=0)
                    with gr.Row():
                        random_seed_button = gr.Button("Randomize Seed")
                        random_seed_button.click(lambda: random.randint(0, 99900), outputs=seed, show_progress="hidden")
                        auto_increment_seed_checkbox = gr.Checkbox(label="Auto Increment Seed", interactive=True, value=True)
                    #num_samples = gr.Number(label="Number of Samples", value=1, minimum=1, maximum=100, precision=0)
                    #batch_size = gr.Number(label="Batch Size", value=1, minimum=1, maximum=100, precision=0)

                # diffusion params

                with gr.Column(min_width=100):
                    num_steps = gr.Number(label="Number of Steps", value=100, minimum=1, maximum=1000, precision=0)
                    cfg_scale = gr.Number(label="CFG Scale", value=1.5, minimum=0, maximum=100, precision=2)
                    use_midpoint = gr.Checkbox(label="Use Midpoint Integration", value=True)
                    num_fgla_iters = gr.Number(label="Number of FGLA Iterations", value=250, minimum=10, maximum=1000, precision=0)
                
                # schedule / noise params

                with gr.Column(min_width=100):
                    #sigma_max = gr.Number(label="Sigma Max", value=self.pipeline.unet.config.sigma_max, minimum=10, maximum=1000, precision=2)
                    #sigma_min = gr.Number(label="Sigma Min", value=self.pipeline.unet.config.sigma_min, minimum=0.01, maximum=1, precision=2)
                    sigma_max = gr.Number(label="Sigma Max", value=200, minimum=10, maximum=1000, precision=2)
                    sigma_min = gr.Number(label="Sigma Min", value=0.15, minimum=0.01, maximum=1, precision=2)
                    rho = gr.Number(label="Rho", value=7, minimum=0.01, maximum=1000, precision=2)
                    input_perturbation = gr.Slider(label="Input Perturbation", minimum=0., maximum=1, step=0.01, value=1)

                # inpainting / img2img params

                with gr.Column(scale=2):
                    input_audio_mode = gr.Radio(label="Input Audio Mode", interactive=True, value="None",
                                                choices=["None", "Img2Img", "Inpaint", "Extend"])
                    img2img_strength = gr.Slider(label="Img2Img Strength", visible=False,
                                                 minimum=0.01, maximum=0.99, step=0.01, value=0.5)
                    with gr.Row():
                        sample_len = self.pipeline.format.config.sample_raw_length / self.pipeline.format.config.sample_rate

                        inpaint_begin = gr.Slider(label="Inpaint Begin (Seconds)", interactive=True, visible=False, minimum=0, maximum=sample_len, step=0.01, value=sample_len/2)
                        inpaint_end = gr.Slider(label="Inpaint End (Seconds)", interactive=True, visible=False, minimum=0, maximum=sample_len, step=0.01, value=sample_len)
                        inpaint_begin.release(lambda begin, end: min(begin, end),
                                              inputs=[inpaint_begin, inpaint_end],
                                              outputs=[inpaint_begin],
                                              show_progress="hidden")
                        inpaint_end.release(lambda begin, end: max(begin, end),
                                            inputs=[inpaint_begin, inpaint_end],
                                            outputs=[inpaint_end],
                                            show_progress="hidden")
                        
                        extend_prepend = gr.Radio(label="Extend Mode", interactive=True, visible=False, value="Extend",
                                                  choices=["Extend", "Prepend"])
                        extend_overlap = gr.Slider(label="Overlap (%)", interactive=True, visible=False, minimum=0, maximum=100, step=0.01, value=50)
                        
                    input_audio = gr.Audio(label="Input Audio", visible=False, type="filepath")

                    def change_input_audio_mode(input_audio_mode):
                        return (gr.update(visible=input_audio_mode == "Img2Img"),
                                gr.update(visible=input_audio_mode == "Inpaint"),
                                gr.update(visible=input_audio_mode == "Inpaint"),
                                gr.update(visible=input_audio_mode == "Extend"),
                                gr.update(visible=input_audio_mode == "Extend"),
                                gr.update(visible=input_audio_mode != "None"))

                    input_audio_mode.change(change_input_audio_mode,
                                            inputs=[input_audio_mode],
                                            outputs=[img2img_strength,
                                                     inpaint_begin, inpaint_end,
                                                     extend_prepend, extend_overlap,
                                                     input_audio],
                                            show_progress="hidden")

            # ********** preset editor **********

            with gr.Row() as self.preset_editor:
                
                preset_files = os.listdir(os.path.join(config.CONFIG_PATH, "sampling", "presets"))
                saved_presets = []
                for file in preset_files:
                    if os.path.splitext(file)[1] == ".json":
                        saved_presets.append(os.path.splitext(file)[0])
                
                preset = gr.Dropdown(
                    choices=saved_presets,
                    label="Select a Preset",
                    value="default",
                    interactive=True,
                    allow_custom_value=True,
                    scale=4,
                )

                save_preset = gr.Button("Save")
                load_preset = gr.Button("Load")


            # ********** prompt editor **********

            prompt = gr.State(value={})

            with gr.Row() as self.prompt_editor:

                game_list = list(f"({self.pipeline.dataset_info['game_train_sample_counts'][game]}) {game}"
                             for game in list(self.pipeline.dataset_game_ids.keys()))
                
                game_dropdown = gr.Dropdown(choices=game_list, label="Select a game", value=game_list[0], scale=4)
                game_weight = gr.Number(label="Weight", value=1, minimum=-100, maximum=100, precision=2)
                add_game_button = gr.Button("Add Game")

            def add_game(prompt, game, weight):
                prompt[game] = weight
                self.logger.debug(f"add_game() prompt state: {prompt}")
                return prompt
            
            def remove_game(prompt, game):
                del prompt[game]
                self.logger.debug(f"remove_game() prompt state: {prompt}")
                return prompt
            
            def change_game(prompt, new_game, old_game, weight):
                del prompt[old_game]
                prompt[new_game] = weight
                self.logger.debug(f"change_game() prompt state: {prompt}")
                return prompt

            add_game_button.click(fn=add_game, inputs=[prompt, game_dropdown, game_weight], outputs=prompt, show_progress="hidden")

            @gr.render(inputs=prompt)
            def render_prompt(prompt_input):

                for prompt_game, prompt_weight in prompt_input.items():
                    with gr.Row():

                        game = gr.Dropdown(choices=game_list, value=prompt_game, show_label=False, scale=4)
                        weight = gr.Number(interactive=True, show_label=False, value=prompt_weight, minimum=-100, maximum=100, precision=2)

                        game.change(change_game, inputs=[prompt, game, gr.State(value=prompt_game), weight], outputs=prompt, show_progress="hidden")
                        weight.change(add_game, inputs=[prompt, game, weight], outputs=prompt, show_progress="hidden")

                        gr.Button("Remove").click(fn=remove_game, inputs=[prompt, game], outputs=prompt, show_progress="hidden")
            
            # ********** sample generation **********

            def get_output_label(seed, num_steps, cfg_scale, sigma_max, sigma_min, rho, input_perturbation,
                                 use_midpoint, num_fgla_iters, img2img_strength, img2img_input, auto_increment_seed):
                params_dict = {"seed": seed, "num_steps": num_steps, "cfg_scale": cfg_scale,
                               "sigma_max": sigma_max, "sigma_min": sigma_min, "rho": rho, "input_perturbation": input_perturbation,
                               "use_midpoint": use_midpoint, "num_fgla_iters": num_fgla_iters, "img2img_strength": img2img_strength}
                
                stripped_prompt = {game.split(" ", 1)[1]: weight for game, weight in prompt.items()}
                if auto_increment_seed == True: seed += 1
                return str(params_dict) + "\n" + str(stripped_prompt), seed
                
            def generate(seed, num_steps, cfg_scale, sigma_max, sigma_min, rho, input_perturbation,
                         use_midpoint, num_fgla_iters, img2img_strength, img2img_input):
                
                stripped_prompt = {game.split(" ", 1)[1]: weight for game, weight in prompt.items()}

                progress = gr.Progress()
                for i in range(num_steps):
                    progress((i+1)/num_steps)
                    time.sleep(0.1)
                    
                latent = np.random.rand(32, 512, 3)
                audio = (np.random.randn(45 * 32000) * 10000).astype(np.int16)
                return latent, (self.pipeline.format.config.sample_rate, audio)

            generate_button = gr.Button("Generate")

            with gr.Group():
                output_label = gr.Textbox(value="", lines=2, max_lines=2, interactive=False, show_label=False)
                latents_output = gr.Image(label="Latents")
                audio_output = gr.Audio(label="Audio", type="numpy")

            generate_button.click(
                fn=get_output_label,
                inputs=[seed, num_steps, cfg_scale, sigma_max, sigma_min, rho, input_perturbation,
                        use_midpoint, num_fgla_iters, img2img_strength, input_audio, auto_increment_seed_checkbox],
                outputs=[output_label, seed],
                show_progress="hidden",
            ).then(
                fn=generate,
                inputs=[seed, num_steps, cfg_scale, sigma_max, sigma_min, rho, input_perturbation,
                        use_midpoint, num_fgla_iters, img2img_strength, input_audio],
                outputs=[latents_output, audio_output],
                concurrency_limit=self.config.gpu_concurrency_limit,
                max_size=self.config.gpu_max_queue_size,
                concurrency_id="gpu",
            )

        with gr.Blocks() as self.log_interface:

            def read_logs():
                if self.log_path is not None:
                    try:
                        with open(self.log_path, "r") as f:
                            return f.read()
                    except Exception as e:
                        self.logger.error(f"Error reading logs at '{self.log_path}': {e}")
                
            with gr.Row():
                logs = gr.Textbox(label="Debug Log", value="", lines=30, max_lines=30, interactive=False, show_copy_button=True)

            self.log_interface.load(read_logs, None, logs, every=1)

        self.generation_interface.queue(default_concurrency_limit=self.config.web_server_default_concurrency_limit,
                                        max_size=self.config.web_server_max_queue_size)

        self.tabbed_interface = gr.TabbedInterface(interface_list=[self.generation_interface, self.settings_interface, self.log_interface],
                                                   title="Dual Diffusion - Generative Diffusion SNES/SFC Music Model",
                                                   tab_names=["Generator", "Model Settings", "Logs"], analytics_enabled=False)
        
        self.tabbed_interface.launch(server_name=self.config.web_server_host,
                                     server_port=self.config.web_server_port,
                                     share=self.config.web_server_share,
                                     show_error=True, debug=True)


if __name__ == "__main__":

    init_cuda()
    GradioApp().run()