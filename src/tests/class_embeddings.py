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

import os

import torch

from pipelines.dual_diffusion_pipeline import DualDiffusionPipeline
from modules.module import DualDiffusionModule
from utils.dual_diffusion_utils import init_cuda, tensor_to_img, save_img


@torch.inference_mode()
def class_embeddings_test() -> None:

    test_params = config.load_json(
        os.path.join(config.CONFIG_PATH, "tests", "class_embeddings.json"))
    
    model_name = test_params["model_name"]
    module_name = test_params["module_name"]
    remove_embedding_mean = test_params["remove_embedding_mean"]
    normalize_embeddings = test_params["normalize_embeddings"]
    device = test_params["device"]

    test_output_path = os.path.join(config.DEBUG_PATH, "class_embeddings") if config.DEBUG_PATH is not None else None
    print("Test output path:", test_output_path)

    model_path = os.path.join(config.MODELS_PATH, model_name)
    model_dtype = torch.float32
    print(f"Loading DualDiffusion model from '{model_path}' (dtype={model_dtype})...")
    pipeline = DualDiffusionPipeline.from_pretrained(model_path,
                                                     torch_dtype=model_dtype,
                                                     device=device,
                                                     load_latest_checkpoints=True,
                                                     load_emas={"unet": "pf_ema_std-0.020.safetensors"})
    module: DualDiffusionModule = getattr(pipeline, module_name)

    num_classes = module.config.label_dim
    class_labels = pipeline.get_class_labels(torch.arange(num_classes, device=device))
    class_embeddings = module.get_class_embeddings(class_labels)

    if remove_embedding_mean:
        class_embeddings -= class_embeddings.mean(dim=0, keepdim=True)
    if normalize_embeddings:
        class_embeddings /= torch.linalg.vector_norm(class_embeddings, dim=1, keepdim=True) / class_embeddings.shape[1]**0.5

    inner_products = torch.einsum('ik,jk->ij', class_embeddings, class_embeddings)
    det = torch.linalg.det(inner_products)
    inner_products.diagonal().zero_()

    game_scores = inner_products.mean(dim=0).tolist()
    game_id_to_name = {v: k for k, v in pipeline.dataset_info["game_id"].items()}
    game_scores = {game_id_to_name[i]: game_scores[i] for i in range(num_classes)}
    game_scores = dict(sorted(game_scores.items(), key=lambda item: item[1]))
    name_padding = max(len(name) for name in game_scores)
    for game, game_score in game_scores.items():
        print(f"{game:<{name_padding}}: {game_score:.4f}")

    print(f"Gram matrix determinant: {det:.4f}")

    if test_output_path is not None:
        inner_products_img_path = os.path.join(test_output_path, "inner_products.png")
        print(f"Saving '{inner_products_img_path}'...")
        save_img(tensor_to_img(inner_products, colormap=True), inner_products_img_path)


if __name__ == "__main__":

    init_cuda()
    class_embeddings_test()