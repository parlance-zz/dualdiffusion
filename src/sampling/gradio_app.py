from utils import config

from typing import Optional
from dataclasses import dataclass
import random
import time
import os

import torch
import numpy as np
import gradio as gr

from utils.dual_diffusion_utils import init_cuda, save_audio, load_audio, dict_str
from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline


@dataclass
class GradioAppConfig:
    model_name: str
    load_latest_checkpoints: bool = True
    load_ema: Optional[str] = None
    device: torch.device = "cuda"
    fp16: bool = True
    compile_params: Optional[dict] = None

    web_server_host: Optional[str] = None
    web_server_port: int = 3001
    web_server_share: bool = False
    web_server_default_concurrency_limit: int = 1
    web_server_max_queue_size: int = 10

class GradioApp:

    def __init__(self) -> None:

        app_config = GradioAppConfig(**config.load_json(
            os.path.join(config.CONFIG_PATH, "sampling", "gradio_app.json")))
        print(f"GradioAppConfig:\n{dict_str(app_config.__dict__)}")

        # load model
        model_dtype = torch.bfloat16 if app_config.fp16 else torch.float32
        model_path = os.path.join(config.MODELS_PATH, app_config.model_name)
        load_emas = {"unet": app_config.load_ema} if app_config.load_ema is not None else None

        print(f"Loading DualDiffusion model from '{model_path}'...")
        pipeline = DualDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=model_dtype, device=app_config.device,
            load_latest_checkpoints=app_config.load_latest_checkpoints, load_emas=load_emas)

        self.pipeline = pipeline
        self.config = app_config

    def run(self) -> None:

        with gr.Blocks() as self.gradio_interface:

            # ********** parameter editor **********

            with gr.Row():

                # general params
                
                with gr.Column(min_width=100):
                    seed = gr.Number(label="Seed", value=42, minimum=0, maximum=99900, precision=0)
                    with gr.Row():
                        random_seed_button = gr.Button("Randomize Seed")
                        random_seed_button.click(lambda: random.randint(0, 99900), outputs=seed, show_progress="hidden")
                        auto_increment_seed_checkbox = gr.Checkbox(label="Auto Increment Seed", interactive=True, value=True)
                    num_samples = gr.Number(label="Number of Samples", value=1, minimum=1, maximum=100, precision=0)
                    batch_size = gr.Number(label="Batch Size", value=1, minimum=1, maximum=100, precision=0)

                with gr.Column(min_width=100):
                    num_steps = gr.Number(label="Number of Steps", value=100, minimum=1, maximum=1000, precision=0)
                    cfg_scale = gr.Number(label="CFG Scale", value=1.5, minimum=0, maximum=100, precision=2)
                    use_midpoint = gr.Checkbox(label="Use Midpoint Integration", value=True)
                    num_fgla_iters = gr.Number(label="Number of FGLA Iterations", value=250, minimum=10, maximum=1000, precision=0)
                    
                with gr.Column(min_width=100):
                    #sigma_max = gr.Number(label="Sigma Max", value=self.pipeline.unet.config.sigma_max, minimum=10, maximum=1000, precision=2)
                    #sigma_min = gr.Number(label="Sigma Min", value=self.pipeline.unet.config.sigma_min, minimum=0.01, maximum=1, precision=2)
                    sigma_max = gr.Number(label="Sigma Max", value=200, minimum=10, maximum=1000, precision=2)
                    sigma_min = gr.Number(label="Sigma Min", value=0.15, minimum=0.01, maximum=1, precision=2)
                    rho = gr.Number(label="Rho", value=7, minimum=0.01, maximum=1000, precision=2)
                    input_perturbation = gr.Slider(label="Input Perturbation", minimum=0., maximum=1, step=0.01, value=1)

                # inpainting / img2img params

                with gr.Column():
                    input_audio_mode = gr.Radio(label="Input Audio Mode", interactive=True, value="None",
                                                choices=["None", "Img2Img", "Inpaint"])
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
                        
                    input_audio = gr.Audio(label="Input Audio", visible=False, type="filepath")

                    def change_input_audio_mode(input_audio_mode):
                        return (gr.update(visible=input_audio_mode == "Img2Img"),
                                gr.update(visible=input_audio_mode == "Inpaint"),
                                gr.update(visible=input_audio_mode == "Inpaint"),
                                gr.update(visible=input_audio_mode != "None"))

                    input_audio_mode.change(change_input_audio_mode,
                                            inputs=[input_audio_mode],
                                            outputs=[img2img_strength, inpaint_begin, inpaint_end, input_audio],
                                            show_progress="hidden")
            
            # ********** prompt editor **********

            with gr.Row():
                game_dropdown = gr.Dropdown(
                    choices=list(self.pipeline.dataset_game_ids.keys()),
                    label="Select a game",
                    value="spc/Gundam Wing - Endless Duel",  
                    scale=4
                )
                game_weight = gr.Number(label="Weight", value=1, minimum=-100, maximum=100, precision=2)
                add_game_button = gr.Button("Add Game")
            
            prompt = {}

            def add_game(game, weight):
                prompt[game] = weight
                return str(prompt)
            
            def remove_game(game):
                del prompt[game]
                return str(prompt)
            
            prompt_textbox = gr.Textbox(label="str(prompt)", interactive=False, visible=False)

            add_game_button.click(
                fn=add_game,
                inputs=[game_dropdown, game_weight],
                outputs=[prompt_textbox],
                show_progress="hidden"
            )

            @gr.render(inputs=prompt_textbox)
            def show_prompt(_):
                for game, weight in prompt.items():
                    with gr.Row():
                        game = gr.Textbox(game, interactive=False, max_lines=1, show_label=False, scale=4)
                        weight = gr.Number(weight, interactive=True, show_label=False, value=weight, minimum=-100, maximum=100, precision=2)
                        weight.change(add_game, inputs=[game, weight], outputs=prompt_textbox, show_progress="hidden")
                        remove_button = gr.Button("Remove")
                        remove_button.click(fn=remove_game, inputs=[game], outputs=[prompt_textbox], show_progress="hidden")
            
            # ********** sample generation **********

            def get_output_label(seed, num_samples, batch_size, num_steps, cfg_scale, sigma_max, sigma_min, rho, input_perturbation,
                                 use_midpoint, num_fgla_iters, img2img_strength, img2img_input, auto_increment_seed):
                params_dict = {"seed": seed, "num_samples": num_samples, "batch_size": batch_size,
                               "num_steps": num_steps, "cfg_scale": cfg_scale,
                               "sigma_max": sigma_max, "sigma_min": sigma_min, "rho": rho, "input_perturbation": input_perturbation,
                               "use_midpoint": use_midpoint, "num_fgla_iters": num_fgla_iters, "img2img_strength": img2img_strength}
                
                if auto_increment_seed == True: seed += 1
                return str(params_dict) + "\n" + str(prompt), seed
                
            def generate(seed, num_samples, batch_size, num_steps, cfg_scale, sigma_max, sigma_min, rho, input_perturbation,
                         use_midpoint, num_fgla_iters, img2img_strength, img2img_input):
                
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
                inputs=[seed, num_samples, batch_size, num_steps, cfg_scale, sigma_max, sigma_min, rho, input_perturbation,
                        use_midpoint, num_fgla_iters, img2img_strength, input_audio, auto_increment_seed_checkbox],
                outputs=[output_label, seed],
                show_progress="hidden",
            ).then(
                fn=generate,
                inputs=[seed, num_samples, batch_size, num_steps, cfg_scale, sigma_max, sigma_min, rho, input_perturbation,
                        use_midpoint, num_fgla_iters, img2img_strength, input_audio],
                outputs=[latents_output, audio_output]
            )

        self.gradio_interface.queue(default_concurrency_limit=self.config.web_server_default_concurrency_limit,
                                    max_size=self.config.web_server_max_queue_size)
        
        self.gradio_interface.launch(server_name=self.config.web_server_host,
                                     server_port=self.config.web_server_port,
                                     share=self.config.web_server_share,
                                     show_error=True, debug=True)


if __name__ == "__main__":

    init_cuda()
    GradioApp().run()