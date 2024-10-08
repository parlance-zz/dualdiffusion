from utils import config

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import threading
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

        self.logger.debug(f"Model metadata:\n{dict_str(pipeline.model_metadata)}")

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

            def get_new_sample(prompt, gen_params):
                return {
                    "name": f"test{gen_params['seed']}",
                    "latents": np.zeros((32*2, 688*2, 3)),
                    "spectrogram": np.zeros((32*8, 688*8, 3)),
                    "audio": (32000, np.zeros(45 * 32000).astype(np.int16)),
                    "prompt": prompt,
                    "gen_params": gen_params,
                    "params_str": f"{dict_str(gen_params)}\n{dict_str(prompt)}"
                }
            
            queued_samples_state_lock = threading.Lock()
            generated_samples_state_lock = threading.Lock()

            def add_sample(prompt, gen_params, output):
                with queued_samples_state_lock:
                    new_sample = get_new_sample(prompt, gen_params)
                    output[new_sample["name"]] = new_sample

                    next_seed = (gen_params["seed"]
                        if gen_params["auto_increment_seed"] == False else gen_params["seed"] + 1)

                    self.logger.debug(f"add_sample() output_state: {output}")
                    return output, next_seed
            
            def remove_queue_sample(output, sample_name):
                with queued_samples_state_lock:                    
                    del output[sample_name]
                    return output
            
            def remove_generated_sample(output, sample_name):
                with generated_samples_state_lock:
                    del output[sample_name]
                    return output

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

            queued_samples_state = gr.State(value={})
            generated_samples_state = gr.State(value={})

            for button in generate_buttons:
                button.click(add_sample, inputs=[prompt_state, gen_param_state, queued_samples_state],
                    outputs=[queued_samples_state, gen_param_components["seed"]], show_progress="hidden")
            
            def render_output_sample(sample, samples_state=None, visible=True, interactive=True):
                components = {
                    "prompt_state": gr.State(value=sample["prompt"]),
                    "gen_params_state": gr.State(value=sample["gen_params"]),
                }

                with gr.Group(visible=visible) as group:
                    with gr.Row():
                        with gr.Column(scale=6):
                            with gr.Row():
                                components["name"] = gr.Textbox(label="Name", value=sample["name"], interactive=False, container=True, show_label=False, lines=1, max_lines=1, scale=6)
                                components["rating"] = gr.Slider(label="Rating", interactive=interactive, container=True, value=0, minimum=0, maximum=10, step=1)
                            
                            components["latents"] = gr.Image(label="Latents", interactive=False, container=False, show_fullscreen_button=False, value=sample["latents"])
                            components["spectrogram"] = gr.Image(label="Spectrogram", interactive=False, container=False, show_fullscreen_button=False, value=sample["spectrogram"])
                            components["audio"] = gr.Audio(label="Audio", interactive=False, type="filepath", container=False, value=sample["audio"])

                        with gr.Column(min_width=50):
                            remove_button = gr.Button("Remove", interactive=interactive)
                            if interactive:
                                remove_button.click(remove_queue_sample if samples_state == queued_samples_state else remove_generated_sample, show_progress="hidden",
                                    inputs=[samples_state, components["name"]], outputs=samples_state)
                            gr.Button("Copy to Audio Input", interactive=interactive)
                            gr.Button("Copy to Params", interactive=interactive)
                            components["params_text"] = gr.Textbox(value=sample["params_str"], interactive=False, container=False,
                                show_label=False, lines=11, max_lines=11, show_copy_button=True)

                return group, components
        
            @gr.render(inputs=queued_samples_state)
            def _(samples):
                for sample in reversed(samples.values()):
                    render_output_sample(sample, samples_state=queued_samples_state, interactive=False)

            generating_sample_group, generating_sample_components = render_output_sample(
                get_new_sample(prompt_state.value, gen_param_state.value), visible=False, interactive=False)
            
            generating_sample_components["status_state"] = gr.State(value="idle")
            #generating_idle_state = gr.State(value=True)
            #start_generation_state = gr.State(value=False)
            #generating_latents_state = gr.State(value=False)
            #decoding_latents_state = gr.State(value=False)
            #synthesizing_waveform_state = gr.State(value=False)
            #generating_sample_components["status_state"].change(lambda status: gr.update(visible=True) if status == "generating" else gr.update(visible=False),
            #    inputs=generating_sample_components["status_state"], outputs=generating_sample_group, show_progress="hidden")

            @gr.render(inputs=generated_samples_state)
            def _(samples):
                for sample in reversed(samples.values()):
                    render_output_sample(sample, samples_state=generated_samples_state, interactive=True)

            def process_queued_samples(queued_samples, status):
                with queued_samples_state_lock:
                    if len(queued_samples) > 0 and status == "idle":
                        queued_samples = {**queued_samples}
                        sample_key = list(queued_samples.keys())[0]
                        sample = queued_samples[sample_key]
                        name = sample["name"]
                        prompt = sample["prompt"]
                        gen_params = sample["gen_params"]
                        rating = 0
                        params_str = sample["params_str"]
                        status = "generating"
                        name += " generating..."
                        del queued_samples[sample_key]
                        return queued_samples, prompt, gen_params, name, rating, params_str, status, gr.update(visible=True)
                    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            #gr.Timer().tick(process_queued_samples, inputs=[queued_samples_state, *list(generating_sample_components.values())],
            #queued_samples_state.change(process_queued_samples, inputs=[queued_samples_state, generating_sample_components["status_state"]],
            gr.Timer().tick(process_queued_samples, inputs=[queued_samples_state, generating_sample_components["status_state"]],
                outputs=[queued_samples_state, generating_sample_components["prompt_state"], generating_sample_components["gen_params_state"], generating_sample_components["name"],
                         generating_sample_components["rating"], generating_sample_components["params_text"], generating_sample_components["status_state"],
                         generating_sample_group], show_progress="hidden",
                concurrency_limit=1, concurrency_id="generation")

            def generate_latents(prompt, gen_params, status):
                
                if status == "idle":
                    return gr.update()
                    
                progress = gr.Progress()
                for i in range(gen_params["num_steps"]):
                    progress((i+1)/gen_params["num_steps"])
                    time.sleep(0.02)
                    latents = np.random.rand(32*2, 688*2, 3)
                    #yield latents

                return latents
            
            def decode_latents(prompt, latents, status):
                
                if status == "idle":
                    return gr.update()
                
                progress = gr.Progress()
                for i in range(100):
                    progress((i+1)/100)
                    time.sleep(0.01)

                spectrogram = np.random.rand(32*8, 688*8, 3)
                return spectrogram

            def synthesize_waveform(prompt, gen_params, name, rating, latents, spectrogram, audio, params_str, status, generated_samples):
                
                if status == "idle":
                    return gr.update(), gr.update()
                
                progress = gr.Progress()
                for i in range(gen_params["num_steps"]):
                    progress((i+1)/gen_params["num_steps"])
                    time.sleep(0.02)

                audio = (32000, (np.random.randn(45 * 32000) * 2000).astype(np.int16))

                sample = {
                    "name": name,
                    "latents": latents,
                    "spectrogram": spectrogram,
                    "audio": audio,
                    "prompt": prompt,
                    "gen_params": gen_params,
                    "params_str": params_str
                }
                with generated_samples_state_lock:
                    generated_samples[name] = sample

                    return audio, generated_samples
            
            def reset_generating_sample(latents, spectrogram):
                latents[:] = 0
                spectrogram[:] = 0
                return "idle", gr.update(visible=False), latents, spectrogram, (32000, np.zeros(45 * 32000).astype(np.int16))
            
            generating_sample_components["status_state"].change(generate_latents,
                inputs=[generating_sample_components["prompt_state"], generating_sample_components["gen_params_state"], generating_sample_components["status_state"]],
                outputs=generating_sample_components["latents"],
                concurrency_limit=1, concurrency_id="generation")
            
            generating_sample_components["latents"].change(decode_latents,
                inputs=[generating_sample_components["prompt_state"], generating_sample_components["latents"], generating_sample_components["status_state"]],
                outputs=generating_sample_components["spectrogram"],
                concurrency_limit=1, concurrency_id="generation")
            
            generating_sample_components["spectrogram"].change(synthesize_waveform,
                inputs=[*list(generating_sample_components.values()), generated_samples_state],
                outputs=[generating_sample_components["audio"], generated_samples_state],
                #outputs=[*list(generating_sample_components.values()), generated_samples_state],
                concurrency_limit=1, concurrency_id="generation")
            
            generating_sample_components["audio"].change(reset_generating_sample,
                inputs=[generating_sample_components["latents"], generating_sample_components["spectrogram"]],
                outputs=[generating_sample_components["status_state"], generating_sample_group, generating_sample_components["latents"], generating_sample_components["spectrogram"], generating_sample_components["audio"]],
                show_progress="hidden", concurrency_limit=1, concurrency_id="generation")

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