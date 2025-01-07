# MIT License
#
# Copyright (c) 2023 Christopher Friesen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from utils import config

import argparse

import torch

from sampling.nicegui_app import NiceGUIApp

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline, SampleParams
from utils.dual_diffusion_utils import init_cuda, save_audio, load_audio, dict_str

@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:

    if args.interactive == True:
        return NiceGUIApp().run()

    if args.sample_cfg_file is None:
        raise ValueError("Must either launch with interactive enabled or provide a sampling configuration file.")
    
    raise NotImplementedError("Non-interactive sampling is not yet implemented.")

    init_cuda()

    # todo: non-interactive sampling scripts

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DualDiffusion sampling script.")
    parser.add_argument(
        "--sample_cfg_file",
        type=str,
        required=False,
        help="Use a preset sampling configuration file in $CONFIG_PATH/sampling/*",
    )
    parser.add_argument(
        "--interactive",
        type=bool,
        default=True,
        help="Start a local web interface for interactive sampling (TBD)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sample(parse_args())