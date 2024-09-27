from utils import config

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import random
import time
import os

import torch
import numpy as np
import gradio as gr

from utils.dual_diffusion_utils import (
    init_cuda, save_audio, load_audio, dict_str,
    get_available_torch_devices, sanitize_filename
)
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline


@dataclass
class GradioAppConfig:
    model_name: str
    model_load_options: dict
    gpu_concurrency_limit: int = 1

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

        with gr.Blocks() as self.log_interface:
            
            last_log_modified_time = gr.State(value=os.path.getmtime(self.log_path)
                                              if self.log_path is not None else 0)
            with gr.Row():
                logs = gr.Textbox(
                    label="Debug Log", value="", lines=20, max_lines=20,
                    interactive=False, show_copy_button=True)
                
            if self.log_path is not None:

                def read_logs(logs, last_log_modified_time):

                    log_modified_time = os.path.getmtime(self.log_path)
                    
                    if log_modified_time > last_log_modified_time:
                        last_log_modified_time = log_modified_time

                        try:
                            with open(self.log_path, "r") as f:
                                logs = f.read()
                        except Exception as e:
                            self.logger.error(f"Error reading logs at '{self.log_path}': {e}")

                    return logs, last_log_modified_time
                            
                self.log_interface.load(
                    read_logs, every=1, show_progress="hidden",
                    inputs=[logs, last_log_modified_time], 
                    outputs=[logs, last_log_modified_time])
            else:
                logs.value = "WARNING: DEBUG_PATH not defined, logging disabled"

        with gr.Blocks() as self.settings_interface:

            with gr.Row():
                
                def get_model_list():
                    return [model_name for model_name in os.listdir(config.MODELS_PATH)
                        if os.path.isdir(os.path.join(config.MODELS_PATH, model_name))]
                
                with gr.Column(scale=4):
                    model_dropdown = gr.Dropdown(
                        choices=get_model_list(), label="Select a model",
                        value=self.config.model_name, interactive=True)
                
                with gr.Column(min_width=50):
                    refresh_model_list_button = gr.Button("Refresh Model List")
                    refresh_model_list_button.click(
                        lambda: gr.update(choices=get_model_list()),
                        outputs=model_dropdown, show_progress="hidden")

                    # todo: load model
                    load_model_button = gr.Button("Load Model")

            with gr.Row():
                
                available_torch_devices = get_available_torch_devices()

                model_path = os.path.join(config.MODELS_PATH, model_dropdown.value)
                model_module_classes = DualDiffusionPipeline.get_model_module_classes(model_path)
                model_module_inventory = DualDiffusionPipeline.get_model_module_inventory(model_path)
                
                for module_name, module_inventory in model_module_inventory.items():
                    with gr.Column():

                        class_name_textbox = gr.Textbox(
                            value=f"Module Class: {model_module_classes[module_name].__name__}",
                            interactive=False, show_label=False, lines=1, max_lines=1)

                        if model_module_classes[module_name].has_trainable_parameters:
                            checkpoint_dropdown = gr.Dropdown(
                                choices=["None"] + module_inventory.checkpoints,
                                label=f"Select a {module_name} Checkpoint",
                                value=(module_inventory.checkpoints[-1]
                                       if len(module_inventory.checkpoints) > 0 else "None"),
                                interactive=True)

                            current_checkpoint = "" if checkpoint_dropdown.value == "None" else checkpoint_dropdown.value
                            ema_dropdown = gr.Dropdown(
                                choices=["None"] + module_inventory.emas[current_checkpoint],
                                label="Select an EMA",
                                value="None",
                                interactive=True)

                        with gr.Row():
                            
                            if model_module_classes[module_name].supports_half_precision:                                
                                fp16_checkbox = gr.Checkbox(label="Use 16-bit Precision", value=True, interactive=True, min_width=50)

                            if model_module_classes[module_name].supports_compile:
                                compile_checkbox = gr.Checkbox(label="Compile Module", value=True, interactive=True, min_width=50)

                            device_dropdown = gr.Dropdown(
                                choices=available_torch_devices,
                                label="Device",
                                value=self.config.model_load_options["device"],
                                interactive=True,
                                min_width=50)
                            
        with gr.Blocks() as self.generation_interface:

            # ********** parameter editor **********

            def get_saved_presets():
                preset_files = os.listdir(os.path.join(config.CONFIG_PATH, "sampling", "presets"))
                saved_presets = []
                for file in preset_files:
                    if os.path.splitext(file)[1] == ".json":
                        saved_presets.append(os.path.splitext(file)[0])

                return sorted(saved_presets)

            saved_presets_state = gr.State(value=get_saved_presets())
            last_loaded_preset_state = gr.State(value="default")
            current_preset_state = gr.State(value="default")
            loading_preset_state = gr.State(value=False)

            prompt_state = gr.State(value={})
            gen_param_components = {}
            gen_param_state = gr.State(value={})
            gen_param_state_modified_state = gr.State(value=False)
            generate_buttons = []

            with gr.Row():
                with gr.Column(scale=2):

                    with gr.Row() as self.gen_param_editor:
                        with gr.Column(min_width=50):
                            gen_param_components["seed"] = gr.Number(label="Seed", value=42, minimum=0, maximum=99900, precision=0, step=1)
                            gen_param_components["auto_increment_seed"] = gr.Checkbox(label="Auto Increment Seed", interactive=True, value=True)
                            with gr.Row():
                                gr.Button("Randomize Seed").click(lambda: random.randint(0, 99900),
                                    outputs=gen_param_components["seed"], show_progress="hidden")
                                generate_buttons += [gr.Button("Generate")]
                        with gr.Column(min_width=50):
                            gen_param_components["num_steps"] = gr.Number(label="Number of Steps", value=100, minimum=10, maximum=1000, precision=0, step=10)
                            gen_param_components["cfg_scale"] = gr.Number(label="CFG Scale", value=1.5, minimum=0, maximum=100, precision=2, step=0.1)
                            gen_param_components["use_heun"] = gr.Checkbox(label="Use Heun's Method", value=True)
                            gen_param_components["num_fgla_iters"] = gr.Number(label="Number of FGLA Iterations", value=250, minimum=50, maximum=1000, precision=0, step=50)
                        with gr.Column(min_width=50):
                            gen_param_components["sigma_max"] = gr.Number(label="Sigma Max", value=200, minimum=10, maximum=1000, precision=2, step=10)
                            gen_param_components["sigma_min"] = gr.Number(label="Sigma Min", value=0.15, minimum=0.05, maximum=2, precision=2, step=0.05)
                            gen_param_components["rho"] = gr.Number(label="Rho", value=7, minimum=0.05, maximum=1000, precision=2, step=0.05)
                            gen_param_components["input_perturbation"] = gr.Slider(label="Input Perturbation", value=1, minimum=0, maximum=1, step=0.01)

                    with gr.Row() as self.preset_editor:
                        
                        def save_preset(preset, saved_presets, prompt, gen_params):
                            preset = sanitize_filename(preset)
                            save_preset_path = os.path.join(
                                config.CONFIG_PATH, "sampling", "presets", f"{preset}.json")
                            config.save_json({"prompt": prompt, "gen_params": gen_params}, save_preset_path)

                            return False, preset, get_saved_presets()
                        
                        def load_preset(preset):
                            preset = sanitize_filename(preset)
                            load_preset_path = os.path.join(
                                config.CONFIG_PATH, "sampling", "presets", f"{preset}.json")
                            loaded_preset_dict = config.load_json(load_preset_path)

                            gen_params = ()
                            for name in gen_param_components.keys():
                                gen_params += (loaded_preset_dict["gen_params"][name],)

                            return (False, True, preset, loaded_preset_dict["prompt"],
                                    loaded_preset_dict["gen_params"]) + gen_params
                        
                        def delete_preset(preset):
                            preset = sanitize_filename(preset)
                            delete_preset_path = os.path.join(
                                config.CONFIG_PATH, "sampling", "presets", f"{preset}.json")
                            os.remove(delete_preset_path)

                            return get_saved_presets()
                        
                        @gr.render(inputs=[saved_presets_state, last_loaded_preset_state,
                            current_preset_state, gen_param_state_modified_state])
                        def _(saved_presets, last_loaded_preset, current_preset, gen_param_state_modified):
                            
                            current_preset = sanitize_filename(current_preset)
                            loaded_preset_label = f"loaded preset: {last_loaded_preset}"
                            if gen_param_state_modified == True: loaded_preset_label += "*"

                            preset_dropdown = gr.Dropdown(
                                choices=saved_presets, label=f"Select a Preset - ({loaded_preset_label})",
                                value=current_preset, interactive=True, allow_custom_value=True, scale=3)
                            preset_dropdown.change(lambda preset: sanitize_filename(preset),
                                inputs=preset_dropdown, outputs=current_preset_state, show_progress="hidden")
                            
                            with gr.Column(min_width=50):

                                def reset_loading_preset_state():
                                    time.sleep(0.3) # gradio pls :(
                                    return False, False
                                
                                save_button_enabled = current_preset != last_loaded_preset or gen_param_state_modified == True
                                save_preset_button = gr.Button("Save Changes", interactive=save_button_enabled)
                                load_button_enabled = save_button_enabled and current_preset in saved_presets
                                load_preset_button = gr.Button("Load Preset", interactive=load_button_enabled)
                                delete_button_enabled = current_preset in saved_presets and current_preset != "default"
                                delete_preset_button = gr.Button("Delete Preset", interactive=delete_button_enabled)

                                save_preset_button.click(save_preset, show_progress="hidden",
                                    inputs=[current_preset_state, saved_presets_state, prompt_state, gen_param_state],
                                    outputs=[gen_param_state_modified_state, last_loaded_preset_state, saved_presets_state])
                                load_preset_button.click(load_preset, inputs=current_preset_state, show_progress="hidden",
                                    outputs=[gen_param_state_modified_state, loading_preset_state, last_loaded_preset_state,
                                prompt_state, gen_param_state] + list(gen_param_components.values())).then(
                                    reset_loading_preset_state, show_progress="hidden",
                                    outputs=[loading_preset_state, gen_param_state_modified_state])
                                delete_preset_button.click(delete_preset, inputs=current_preset_state,
                                    outputs=saved_presets_state, show_progress="hidden")
                        
                # inpainting / img2img params
                with gr.Column() as self.input_audio_editor:
                    input_audio_mode = gr.Radio(label="Audio Input Mode", interactive=True,
                        value="None", choices=["None", "Img2Img", "Inpaint", "Outpaint"])

                    with gr.Row():
                        #todo: sample_len needs to come from selected input audio
                        sample_len = (self.pipeline.format.config.sample_raw_length
                                      / self.pipeline.format.config.sample_rate)

                        img2img_strength = gr.Slider(label="Img2Img Strength", interactive=True,
                            visible=False, minimum=0.01, maximum=0.99, step=0.01, value=0.5)
                        
                        inpaint_begin = gr.Slider(label="Inpaint Begin (Seconds)", interactive=True,
                            visible=False, minimum=0, maximum=sample_len, step=0.1, value=sample_len/2)
                        inpaint_end = gr.Slider(label="Inpaint End (Seconds)", interactive=True,
                            visible=False, minimum=0, maximum=sample_len, step=0.1, value=sample_len)
                        inpaint_begin.release(lambda begin, end: min(begin, end),
                            inputs=[inpaint_begin, inpaint_end], outputs=[inpaint_begin], show_progress="hidden")
                        inpaint_end.release(lambda begin, end: max(begin, end),
                            inputs=[inpaint_begin, inpaint_end], outputs=[inpaint_end], show_progress="hidden")
                        
                        extend_prepend = gr.Radio(label="Outpaint Mode", interactive=True,
                            value="Extend", choices=["Extend", "Prepend"], visible=False)
                        extend_overlap = gr.Slider(label="Overlap (%)", interactive=True,
                            minimum=0, maximum=100, step=1, value=50, visible=False)
                    
                    input_audio = gr.Audio(label="Input Audio", visible=False, type="filepath", interactive=True, editable=True)

                    def change_input_audio_mode(input_audio_mode):
                        return (gr.update(visible=input_audio_mode == "Img2Img"),
                                gr.update(visible=input_audio_mode == "Inpaint"),
                                gr.update(visible=input_audio_mode == "Inpaint"),
                                gr.update(visible=input_audio_mode == "Outpaint"),
                                gr.update(visible=input_audio_mode == "Outpaint"),
                                gr.update(visible=input_audio_mode != "None"))

                    input_audio_mode.change(change_input_audio_mode, inputs=[input_audio_mode],
                        outputs=[img2img_strength, inpaint_begin, inpaint_end, extend_prepend,
                                 extend_overlap, input_audio], show_progress="hidden")

            def update_gen_param_state(loading_preset, gen_param_state, name, component, gen_param_state_modified):
                gen_param_state[name] = component
                self.logger.debug(f"update_gen_param_state() gen_param_state: {gen_param_state}")

                if name not in ["seed", "auto_increment_seed"]:
                    gen_param_state_modified = loading_preset == False
                return gen_param_state, gen_param_state_modified
               
            for name, component in gen_param_components.items():
                component.change(update_gen_param_state,
                    inputs=[loading_preset_state, gen_param_state, gr.State(value=name), component, gen_param_state_modified_state],
                    outputs=[gen_param_state, gen_param_state_modified_state], show_progress="hidden")
                gen_param_state.value[name] = component.value
            
            self.logger.debug(f"initial gen_param_state: {gen_param_state.value}")

            def add_sample(prompt, gen_params, output):
                
                sample_name = f"test{gen_params['seed']}"
                if not sample_name in output:
                    output[sample_name] = {}

                output[sample_name]["latents"] = np.zeros((32*2, 688*2, 3))
                output[sample_name]["spectrogram"] = np.zeros((32*8, 688*8, 3))
                output[sample_name]["audio"] = 32000, np.zeros(45 * 32000).astype(np.int16)
                output[sample_name]["prompt"] = prompt
                output[sample_name]["gen_params"] = gen_params
                output[sample_name]["params_str"] = f"{dict_str(gen_params)}\n{dict_str(prompt)}"

                next_seed = (gen_params["seed"]
                    if gen_params["auto_increment_seed"] == False else gen_params["seed"] + 1)

                self.logger.debug(f"add_sample() output_state: {output}")

                #progress = gr.Progress()
                #for i in range(gen_params["num_steps"]):
                #    progress((i+1)/gen_params["num_steps"])
                #    time.sleep(0.03)
                    
                return output, next_seed
            
            def remove_sample(output, sample_name):
                del output[sample_name]
                return output
            
            def generate_latents(output):
                
                for sample_name, sample in output.items():
                    if sample["status"] == "queued":
                        
                        
                        prompt = output["prompt"]
                        gen_params = output["gen_params"]
                        #input_audio = output["input_audio"]

                        progress = gr.Progress()
                        for i in range(gen_params["num_steps"]):
                            progress((i+1)/gen_params["num_steps"])
                            time.sleep(0.03)
                            latents = np.random.randn((32*2, 688*2, 3))
                            yield latents

                        sample["status"] = "generating_latents"

                        return output, latents
            
            def decode_latents(output):
                pass

            def synthesize_waveform(output):
                pass

            # ********** prompt editor **********

            with gr.Group():
                with gr.Row():
                    game_list = list((f"({self.pipeline.dataset_info['game_train_sample_counts'][game]}) {game}", game)
                                    for game in list(self.pipeline.dataset_game_ids.keys()))
                    
                    game_dropdown = gr.Dropdown(choices=game_list, label="Select a game", value=game_list[0][1], scale=8)
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

            add_game_button.click(fn=add_game, inputs=[prompt_state, game_dropdown, game_weight],
                outputs=prompt_state, show_progress="hidden")
            prompt_state.change(lambda loading_preset: loading_preset == False, inputs=loading_preset_state,
                outputs=gen_param_state_modified_state, show_progress="hidden")

            @gr.render(inputs=prompt_state)
            def _(prompt_input):
                rendered_label = False
                with gr.Group():
                    for prompt_game, prompt_weight in prompt_input.items():
                        with gr.Row():
                            game = gr.Dropdown(choices=game_list, show_label=(rendered_label == False),
                                value=prompt_game, label="Selected Games:", scale=8)
                            weight = gr.Number(interactive=True, show_label=(rendered_label == False),
                                value=prompt_weight, minimum=-100, maximum=100, precision=2, label="Weight:", min_width=50)
                            
                            game.change(change_game, inputs=[prompt_state, game, gr.State(value=prompt_game), weight],
                                outputs=prompt_state, show_progress="hidden")
                            weight.change(add_game, inputs=[prompt_state, game, weight],
                                outputs=prompt_state, show_progress="hidden")

                            if rendered_label == True:
                                gr.Button("Remove").click(fn=remove_game, inputs=[prompt_state, game],
                                    outputs=prompt_state, show_progress="hidden")
                            else: gr.Button("Generate")
                            rendered_label = True
            
            # ********** sample generation **********

            output_state = gr.State(value={})

            for button in generate_buttons:
                button.click(add_sample, inputs=[prompt_state, gen_param_state, output_state],
                    outputs=[output_state, gen_param_components["seed"]], show_progress="hidden")
            
            @gr.render(inputs=output_state)
            def _(output):

                for sample_name, sample in reversed(output.items()):
                    
                    with gr.Group():
                        with gr.Row():
                            with gr.Column(scale=6):
                                sample_name_state = gr.State(value=sample_name)

                                with gr.Row():
                                    gr.Textbox(label="Name", value=sample_name, interactive=False, container=True, show_label=False, lines=1, max_lines=1, scale=6)
                                    gr.Slider(label="Rating", interactive=True, container=True, value=0, minimum=0, maximum=10, step=1)
                                
                                gr.Image(label="Latents", interactive=False, container=False, show_fullscreen_button=False, value=sample["latents"])
                                gr.Image(label="Spectrogram", interactive=False, container=False, show_fullscreen_button=False, value=sample["spectrogram"])
                                gr.Audio(label="Audio", interactive=False, type="filepath", container=False, value=sample["audio"])

                            with gr.Column(min_width=50):
                                gr.Button("Remove").click(remove_sample, show_progress="hidden",
                                    inputs=[output_state, sample_name_state], outputs=output_state)
                                gr.Button("Copy to Audio Input")
                                gr.Button("Copy to Params")
                                gr.Textbox(value=sample["params_str"], interactive=False, container=False,
                                    show_label=False, lines=11, max_lines=11, show_copy_button=True)


            """
            generate_button.click(
                fn=get_output_label,
                inputs=[prompt_state, seed, num_steps, cfg_scale, sigma_max, sigma_min, rho, input_perturbation,
                        use_midpoint, num_fgla_iters, img2img_strength, input_audio, auto_increment_seed_checkbox],
                outputs=[output_label, seed],
                show_progress="hidden",
            ).then(
                fn=generate,
                inputs=[prompt_state, seed, num_steps, cfg_scale, sigma_max, sigma_min, rho, input_perturbation,
                        use_midpoint, num_fgla_iters, img2img_strength, input_audio],
                outputs=[latents_output, audio_output],
                concurrency_limit=self.config.gpu_concurrency_limit,
                concurrency_id="gpu",
            )
            """

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