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

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.autoencoder_kl import AutoencoderKLOutput

from unet2d_dual_blocks import SeparableAttnDownBlock2D, SeparableAttnUpBlock2D, SeparableMidBlock2D
from dual_diffusion_utils import mdct, get_activation, save_raw, hz_to_mels, mels_to_hz, get_mel_density, get_hann_window, get_kaiser_window, ScaleNorm

class DualMultiscaleSpectralLoss4:

    @torch.no_grad()
    def __init__(self, loss_params):

        self.edge_crop_width = loss_params["edge_crop_width"]
        self.sample_block_width = loss_params["sample_block_width"]
        self.num_filters = loss_params["num_filters"]
        self.min_freq = loss_params["min_freq"]
        self.max_freq = loss_params["max_freq"]
        self.min_logvar = loss_params["min_logvar"]
        self.max_logvar = loss_params["max_logvar"]
        self.sample_rate = loss_params["sample_rate"]
        self.u = loss_params["u"]

        self.filters = None
        self.write_debug = True

    @torch.no_grad()
    def create_filters(self, crop_width, device="cpu"):
        
        if self.filters is None:
            self.filters = torch.zeros((1, self.num_filters, crop_width), device=device)
        else:
            if self.filters.shape[-1] != crop_width:
                self.filters = torch.zeros((1, self.num_filters, crop_width), device=device)

            if self.filters.device != device:
                self.filters = self.filters.to(device)

            self.write_debug = False

        # linear frequency spacing
        filter_hz = torch.rand(self.num_filters, device=device) * (self.max_freq - self.min_freq) + self.min_freq
        filter_mels = hz_to_mels(filter_hz)
        
        # mel scale spacing
        #min_mels = hz_to_mels(torch.tensor(self.min_freq, device=device))
        #max_mels = hz_to_mels(torch.tensor(self.max_freq, device=device))
        #filter_mels = torch.rand(self.num_filters, device=device) * (max_mels - min_mels) + min_mels

        filter_logvar = torch.rand(self.num_filters, device=device) * (self.max_logvar - self.min_logvar) + self.min_logvar
        filter_var = torch.exp2(filter_logvar)

        fft_q = torch.arange(0, crop_width // 2 + 1, device=device) / (crop_width // 2)
        fft_mels = hz_to_mels(fft_q * self.sample_rate)

        self.filters[0, :, :crop_width//2 + 1] = torch.exp(-(filter_mels.view(-1, 1) - fft_mels.view(1, -1)).square() / filter_var.view(-1, 1))
        self.filters /= self.filters.square().mean(dim=-1, keepdim=True)

        # these 2 normalizations are ~= to the power norm above
        #self.filters /= filter_var.sqrt().view(1, -1, 1) / np.exp2(self.max_logvar / 2)
        #self.filters *= get_mel_density(filter_hz).view(1, -1, 1)

        #self.filters[0, :, :crop_width//2 + 1] = torch.exp(-torch.log(filter_q.view(-1, 1) / fft_q.view(1, -1)).square() / filter_var.view(-1, 1))
        #self.filters[0, :, 0] = 0
        #self.filters /= self.filters.square().mean(dim=-1, keepdim=True)

        if self.write_debug:
            debug_path = os.environ.get("DEBUG_PATH", None)
            if debug_path is not None:
                save_raw(self.filters.abs(), os.path.join(debug_path, "debug_multiscale_spectral_loss_filters.raw"))
                save_raw(self.filters.abs().mean(dim=1)[0, :crop_width//2+1], os.path.join(debug_path, "debug_multiscale_spectral_loss_filter_coverage.raw"))
                save_raw(torch.fft.fftshift(torch.fft.ifft(self.filters, norm="ortho"), dim=-1), os.path.join(debug_path, "debug_multiscale_spectral_loss_filters_ifft.raw"))

    def __call__(self, sample, target):
        
        target = target[:, self.sample_block_width // 2:-self.sample_block_width]

        if sample.shape != target.shape:
            raise ValueError(f"sample.shape != target.shape. sample.shape: {sample.shape}. target.shape: {target.shape}.")

        bsz = sample.shape[0]
        self.create_filters(sample.shape[-1], device=sample.device)

        sample_fft = torch.fft.fft(sample, norm="forward") * 1024
        sample_filtered_abs = torch.fft.ifft(sample_fft.view(bsz, 1, -1) * self.filters, norm="backward").abs()
        if self.edge_crop_width > 0:
            sample_filtered_abs = sample_filtered_abs[:, :, self.edge_crop_width:-self.edge_crop_width]
        #sample_filtered_abs = sample_filtered_abs / sample_filtered_abs.amax(dim=(1, 2), keepdim=True)
        #print("sample_filtered_abs_amax: ", sample_filtered_abs.amax(dim=(1, 2), keepdim=True))
        sample_filtered_abs_ln = (sample_filtered_abs * self.u).log1p() / np.log(self.u + 1)

        target_fft = torch.fft.fft(target, norm="forward") * 1024
        target_filtered_abs = torch.fft.ifft(target_fft.view(bsz, 1, -1) * self.filters, norm="backward").abs()
        if self.edge_crop_width > 0:
            target_filtered_abs = target_filtered_abs[:, :, self.edge_crop_width:-self.edge_crop_width]
        #target_filtered_abs = target_filtered_abs / target_filtered_abs.amax(dim=(1,2), keepdim=True)
        #print("target_filtered_abs_amax: ", target_filtered_abs.amax(dim=(1, 2), keepdim=True))
        target_filtered_abs_ln = (target_filtered_abs * self.u).log1p() / np.log(self.u + 1)

        if self.write_debug:
            debug_path = os.environ.get("DEBUG_PATH", None)
            if debug_path is not None:
                with torch.no_grad():
                    target_filtered = torch.fft.ifft(target_fft.view(bsz, 1, -1) * self.filters, norm="backward")
                    save_raw(target_filtered, os.path.join(debug_path, "debug_multiscale_spectral_loss_target_filtered.raw"))
                    target_reconstructed = target_filtered.mean(dim=1)
                    save_raw(target_reconstructed, os.path.join(debug_path, "debug_multiscale_spectral_loss_target_reconstructed.raw"))
                    save_raw(target, os.path.join(debug_path, "debug_multiscale_spectral_loss_target.raw"))
                torch.cuda.empty_cache()
            self.write_debug = False             
        
        return torch.nn.functional.mse_loss(sample_filtered_abs_ln, target_filtered_abs_ln, reduction="mean") * 10
    
class DualMultiscaleSpectralLoss3:

    @torch.no_grad()
    def __init__(self, loss_params):

        self.edge_crop_width = loss_params["edge_crop_width"]
        self.sample_block_width = loss_params["sample_block_width"]
        self.num_filters = loss_params["num_filters"]
        self.min_freq = loss_params["min_freq"]
        self.max_freq = loss_params["max_freq"]
        self.min_std = loss_params["min_std"]
        self.std_octaves = loss_params["std_octaves"]
        self.u = loss_params["u"]

        self.filters = None
        self.write_debug = False

    @torch.no_grad()
    def create_filters(self, crop_width, device="cpu"):
        
        if self.filters is None:
            self.filters = torch.zeros((1, self.num_filters, crop_width), device=device)
        else:
            if self.filters.shape[-1] != crop_width:
                self.filters = torch.zeros((1, self.num_filters, crop_width), device=device)
            if self.filters.device != device:
                self.filters = self.filters.to(device)

            self.write_debug = False

        num_filter_octaves = -np.log2(self.min_freq / self.max_freq)
        filter_q = torch.exp2(-torch.rand(self.num_filters, device=device) * num_filter_octaves) * self.max_freq
        fft_q = torch.arange(0, crop_width // 2 + 1, device=device) / (crop_width // 2)
        
        filter_std = self.min_std * torch.exp2(torch.rand(self.num_filters, device=device) * self.std_octaves)
    
        self.filters[0, :, :crop_width//2 + 1] = torch.exp(-filter_std.view(-1, 1) * torch.log(filter_q.view(-1, 1) / fft_q.view(1, -1)).square())
        self.filters[0, :, 0] = 0

        if self.write_debug:
            debug_path = os.environ.get("DEBUG_PATH", None)
            if debug_path is not None:
                save_raw(self.filters.abs(), os.path.join(debug_path, "debug_multiscale_spectral_loss_filters.raw"))
                save_raw(self.filters.abs().mean(dim=1), os.path.join(debug_path, "debug_multiscale_spectral_loss_filter_coverage.raw"))
                save_raw(torch.fft.fftshift(torch.fft.ifft(self.filters, norm="ortho"), dim=-1), os.path.join(debug_path, "debug_multiscale_spectral_loss_filters_ifft.raw"))

    def __call__(self, sample, target):
        
        target = target[:, self.sample_block_width // 2:-self.sample_block_width]

        if sample.shape != target.shape:
            raise ValueError(f"sample.shape != target.shape. sample.shape: {sample.shape}. target.shape: {target.shape}.")

        bsz = sample.shape[0]
        self.create_filters(sample.shape[-1], device=sample.device)

        sample_fft = torch.fft.fft(sample, norm="ortho")
        sample_filtered_abs = torch.fft.ifft(sample_fft.view(bsz, 1, -1) * self.filters, norm="ortho").abs()
        if self.edge_crop_width > 0:
            sample_filtered_abs = sample_filtered_abs[:, :, self.edge_crop_width:-self.edge_crop_width]
        sample_filtered_abs = sample_filtered_abs / sample_filtered_abs.amax(dim=(1, 2), keepdim=True)
        sample_filtered_abs_ln = (sample_filtered_abs * self.u).log1p() / np.log(self.u + 1)

        target_fft = torch.fft.fft(target, norm="ortho")
        target_filtered_abs = torch.fft.ifft(target_fft.view(bsz, 1, -1) * self.filters, norm="ortho").abs()
        if self.edge_crop_width > 0:
            target_filtered_abs = target_filtered_abs[:, :, self.edge_crop_width:-self.edge_crop_width]
        target_filtered_abs = target_filtered_abs / target_filtered_abs.amax(dim=(1,2), keepdim=True)
        target_filtered_abs_ln = (target_filtered_abs * self.u).log1p() / np.log(self.u + 1)

        if self.write_debug:
            debug_path = os.environ.get("DEBUG_PATH", None)
            if debug_path is not None:
                with torch.no_grad():
                    target_filtered = torch.fft.ifft(target_fft.view(bsz, 1, -1) * self.filters, norm="ortho")
                    save_raw(target_filtered, os.path.join(debug_path, "debug_multiscale_spectral_loss_target_filtered.raw"))
                torch.cuda.empty_cache()
            self.write_debug = False             
        
        return torch.nn.functional.mse_loss(sample_filtered_abs_ln, target_filtered_abs_ln, reduction="mean") * 8

class DualMultiscaleSpectralLoss2:

    @torch.no_grad()
    def __init__(self, loss_params):

        self.edge_crop_width = loss_params["edge_crop_width"]
        self.sample_block_width = loss_params["sample_block_width"]
        self.num_filters = loss_params["num_filters"]
        self.min_freq = loss_params["min_freq"]
        self.max_freq = loss_params["max_freq"]
        self.std_octaves = loss_params["std_octaves"]
        self.u = loss_params["u"]

        self.filters = None
        self.write_debug = False

    @torch.no_grad()
    def create_filters(self, crop_width, device="cpu"):
        
        if self.filters is None:
            self.filters = torch.zeros((1, self.num_filters, crop_width), device=device)
        else:
            if self.filters.shape[-1] != crop_width:
                self.filters = torch.zeros((1, self.num_filters, crop_width), device=device)

            if self.filters.device != device:
                self.filters = self.filters.to(device)

            self.write_debug = False

        filter_q = torch.rand(self.num_filters, device=device) * (self.max_freq - self.min_freq) + self.min_freq
        filter_std = 2 * torch.exp2(torch.rand(self.num_filters, device=device) * self.std_octaves)
        filter_std = filter_std * filter_q / self.min_freq
        fft_q = torch.arange(0, crop_width // 2 + 1, device=device) / (crop_width // 2)

        self.filters[0, :, :crop_width//2 + 1] = torch.exp(-filter_std.view(-1, 1) * torch.log(filter_q.view(-1, 1) / fft_q.view(1, -1)).square())
        self.filters[0, :, 0] = 0
        self.filters /= filter_q.sqrt().view(1, -1, 1)

        if self.write_debug:
            debug_path = os.environ.get("DEBUG_PATH", None)
            if debug_path is not None:
                save_raw(self.filters.abs(), os.path.join(debug_path, "debug_multiscale_spectral_loss_filters.raw"))
                save_raw(self.filters.abs().mean(dim=1), os.path.join(debug_path, "debug_multiscale_spectral_loss_filter_coverage.raw"))
                save_raw(torch.fft.fftshift(torch.fft.ifft(self.filters, norm="ortho"), dim=-1), os.path.join(debug_path, "debug_multiscale_spectral_loss_filters_ifft.raw"))

    def __call__(self, sample, target):
        
        target = target[:, self.sample_block_width // 2:-self.sample_block_width]

        if sample.shape != target.shape:
            raise ValueError(f"sample.shape != target.shape. sample.shape: {sample.shape}. target.shape: {target.shape}.")

        bsz = sample.shape[0]
        self.create_filters(sample.shape[-1], device=sample.device)

        sample_fft = torch.fft.fft(sample, norm="ortho")
        sample_filtered_abs = torch.fft.ifft(sample_fft.view(bsz, 1, -1) * self.filters, norm="ortho").abs()
        if self.edge_crop_width > 0:
            sample_filtered_abs = sample_filtered_abs[:, :, self.edge_crop_width:-self.edge_crop_width]
        sample_filtered_abs = sample_filtered_abs / sample_filtered_abs.amax(dim=(1, 2), keepdim=True)
        sample_filtered_abs_ln = (sample_filtered_abs * self.u).log1p() / np.log(self.u + 1)

        target_fft = torch.fft.fft(target, norm="ortho")
        target_filtered_abs = torch.fft.ifft(target_fft.view(bsz, 1, -1) * self.filters, norm="ortho").abs()
        if self.edge_crop_width > 0:
            target_filtered_abs = target_filtered_abs[:, :, self.edge_crop_width:-self.edge_crop_width]
        target_filtered_abs = target_filtered_abs / target_filtered_abs.amax(dim=(1,2), keepdim=True)
        target_filtered_abs_ln = (target_filtered_abs * self.u).log1p() / np.log(self.u + 1)

        if self.write_debug:
            debug_path = os.environ.get("DEBUG_PATH", None)
            if debug_path is not None:
                with torch.no_grad():
                    target_filtered = torch.fft.ifft(target_fft.view(bsz, 1, -1) * self.filters, norm="ortho")
                    save_raw(target_filtered, os.path.join(debug_path, "debug_multiscale_spectral_loss_target_filtered.raw"))
                torch.cuda.empty_cache()
            self.write_debug = False             
        
        return torch.nn.functional.mse_loss(sample_filtered_abs_ln, target_filtered_abs_ln, reduction="mean") * 8
    
class DualMultiscaleSpectralLoss:

    @torch.no_grad()
    def __init__(self, loss_params):
    
        self.sample_rate = loss_params["sample_rate"]
        self.sample_block_width = loss_params["sample_block_width"]
        self.block_widths = loss_params["block_widths"]
        self.block_offsets = loss_params["block_offsets"]
        self.u = loss_params["u"]
        self.block_octaves = loss_params["block_octaves"]
        self.sigma = loss_params["sigma"]
        
        #"""
        if isinstance(self.sigma, Tuple):
            if len(self.sigma) != len(self.block_widths):
                raise ValueError(f"Must provide the same number of `sigma` as `block_widths`. `sigma`: {self.sigma}. `block_widths`: {self.block_widths}.")
        else:
            self.sigma = (self.sigma,) * len(self.block_widths)

        if len(self.block_octaves) != len(self.block_widths):
            raise ValueError(f"Must provide the same number of `block_octaves` as `block_widths`. `block_octaves`: {self.block_octaves}. `block_widths`: {self.block_widths}.")
        
        self.block_weights = []
        for block_num, block_width in enumerate(self.block_widths):
      
            block_q = 2 ** -self.block_octaves[block_num]
            n_bins = block_width // 2
            mdct_q = torch.arange(0.5, n_bins + 0.5) / n_bins
            sigma = self.sigma[block_num]

            filter = torch.exp(-sigma * torch.log(mdct_q / block_q).square())
            self.block_weights.append(filter.view(1, 1, -1))
        #"""
        
        #self.abs_loss_scale = torch.log(torch.tensor(self.u, device=target.device))

        self.loss_scale = 1 / (len(self.block_widths) * len(self.block_offsets))
        #self.loss_scale = 1 / (len(self.block_widths) * len(self.block_offsets) * 2)

    def __call__(self, sample, target):
        
        target = target[:, self.sample_block_width // 2:-self.sample_block_width]
        assert(sample.shape == target.shape)
        
        if self.block_weights[0].device != target.device:
            for block_num, block_weight in enumerate(self.block_weights):
                self.block_weights[block_num] = block_weight.to(target.device)

        loss = torch.zeros(1, device=target.device)

        for block_num, block_width in enumerate(self.block_widths):
            for block_offset in self.block_offsets:

                offset = int(block_offset * block_width)
                block_weight = self.block_weights[block_num]

                #sample_fft_abs = mdct(sample[:, offset:], block_width, window_degree=2)[:, 1:-2, :].abs()
                #sample_fft_abs_ln = (sample_fft_abs * self.u).log1p() * block_weight
                sample_fft = mdct(sample[:, offset:], block_width, window_degree=2)[:, 1:-2, :]
                sample_fft = sample_fft / sample_fft.abs().amax(dim=(1,2), keepdim=True)
                sample_fft_response = (sample_fft * block_weight).sum(dim=-1, keepdim=True).abs()
                sample_fft_abs_ln = (sample_fft_response * self.u).log1p()

                #target_fft_abs = mdct(target[:, offset:], block_width, window_degree=2)[:, 1:-2, :].abs()
                #target_fft_abs_ln = (target_fft_abs * self.u).log1p() * block_weight
                target_fft = mdct(target[:, offset:], block_width, window_degree=2)[:, 1:-2, :]
                target_fft = target_fft / target_fft.abs().amax(dim=(1,2), keepdim=True)
                target_fft_response = (target_fft * block_weight).sum(dim=-1, keepdim=True).abs()
                target_fft_abs_ln = (target_fft_response * self.u).log1p()
                
                loss += torch.nn.functional.l1_loss(sample_fft_abs_ln, target_fft_abs_ln, reduction="mean")
                #loss += torch.nn.functional.l1_loss(sample_fft_abs, target_fft_abs, reduction="mean") * self.abs_loss_scale

        return loss * self.loss_scale
    
class EncoderDual(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=4,
        act_fn="silu",
        double_z=True,
        block_out_channels=(128,256,512,512,),
        layers_per_block=2,
        layers_per_mid_block: int = 1,
        norm_num_groups: Union[int, Tuple[int]] = 32,
        norm_eps: float = 1e-6,
        downsample_type: str = "conv",
        add_mid_attention: bool = True,
        attention_num_heads: Union[int, Tuple[int]] = (8,16,32,32),
        separate_attn_dim_down: Tuple[int] = (3,3),
        separate_attn_dim_mid: Tuple[int] = (0,),
        double_attention: Union[bool, Tuple[bool]] = False,
        pre_attention: Union[bool, Tuple[bool]] = False,
        conv_size = (3,3),
        freq_embedding_dim: Union[int, Tuple[int]] = 0,
        time_embedding_dim: Union[int, Tuple[int]] = 0,
        add_attention: Union[bool, Tuple[bool]] = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            if is_final_block is True: _downsample_type = None
            else: _downsample_type = downsample_type or "conv"
            _norm_num_groups = norm_num_groups[i]
            _attention_num_heads = attention_num_heads[i]
            _double_attention = double_attention[i]
            _pre_attention = pre_attention[i]
            _freq_embedding_dim = freq_embedding_dim[i]
            _time_embedding_dim = time_embedding_dim[i]
            _add_attention = add_attention[i]

            down_block = SeparableAttnDownBlock2D(
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=_norm_num_groups,
                attention_num_heads=_attention_num_heads,
                downsample_padding=0, #?
                downsample_type=_downsample_type,
                separate_attn_dim=separate_attn_dim_down,
                double_attention=_double_attention,
                pre_attention=_pre_attention,
                conv_size=conv_size,
                freq_embedding_dim=_freq_embedding_dim,
                time_embedding_dim=_time_embedding_dim,
                add_attention=_add_attention,
                return_skip_samples=False,
                temb_channels=None,
            )

            self.down_blocks.append(down_block)

        # mid
        _norm_num_groups = norm_num_groups[-1]
        _attention_num_heads = attention_num_heads[-1]
        _double_attention = double_attention[-1]
        _pre_attention = pre_attention[-1]
        _freq_embedding_dim = freq_embedding_dim[-1]
        _time_embedding_dim = time_embedding_dim[-1]
        
        self.mid_block = SeparableMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_time_scale_shift="default",
            attention_num_heads=_attention_num_heads,
            resnet_groups=_norm_num_groups,
            add_attention=add_mid_attention,
            pre_attention=_pre_attention,
            separate_attn_dim=separate_attn_dim_mid,
            double_attention=_double_attention,
            conv_size=conv_size,
            num_layers=layers_per_mid_block,
            freq_embedding_dim=_freq_embedding_dim,
            time_embedding_dim=_time_embedding_dim,
        )

        # out
        if norm_num_groups[-1] != 0:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups[-1], eps=norm_eps)
        else:
            self.conv_norm_out = nn.Identity()
        self.conv_act = get_activation(act_fn)

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            for down_block in self.down_blocks:
                sample, _ = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), sample, use_reentrant=False
                )
            # middle
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), sample, use_reentrant=False
            )

        else:
            # down
            for down_block in self.down_blocks:
                sample, _ = down_block(sample)

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

class DecoderDual(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=2,
        act_fn="silu",
        block_out_channels=(128,256,512,512,),
        layers_per_block=2,
        layers_per_mid_block: int = 1,
        norm_num_groups: Union[int, Tuple[int]] = 32,
        norm_eps: float = 1e-6,
        upsample_type: str = "conv",
        add_mid_attention: bool = True,
        attention_num_heads: Union[int, Tuple[int]] = (8,16,32,32),
        separate_attn_dim_up: Tuple[int] = (3,3,3),
        separate_attn_dim_mid: Tuple[int] = (0,),
        double_attention: Union[bool, Tuple[bool]] = False,
        pre_attention: Union[bool, Tuple[bool]] = False,
        conv_size = (3,3),
        freq_embedding_dim: Union[int, Tuple[int]] = 0,
        time_embedding_dim: Union[int, Tuple[int]] = 0,
        add_attention: Union[bool, Tuple[bool]] = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        _norm_num_groups = norm_num_groups[-1]
        _attention_num_heads = attention_num_heads[-1]
        _double_attention = double_attention[-1]
        _pre_attention = pre_attention[-1]
        _freq_embedding_dim = freq_embedding_dim[-1]
        _time_embedding_dim = time_embedding_dim[-1]
        
        self.mid_block = SeparableMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_num_heads=_attention_num_heads,
            resnet_groups=_norm_num_groups,
            add_attention=add_mid_attention,
            pre_attention=_pre_attention,
            separate_attn_dim=separate_attn_dim_mid,
            double_attention=_double_attention,
            conv_size=conv_size,
            num_layers=layers_per_mid_block,
            freq_embedding_dim=_freq_embedding_dim,
            time_embedding_dim=_time_embedding_dim,
        )

        reversed_attention_num_heads = list(reversed(attention_num_heads))
        reversed_norm_num_groups = list(reversed(norm_num_groups))
        reversed_double_attention = list(reversed(double_attention))
        reversed_pre_attention = list(reversed(pre_attention))
        reversed_freq_embedding_dim = list(reversed(freq_embedding_dim))
        reversed_time_embedding_dim = list(reversed(time_embedding_dim))
        reversed_add_attention = list(reversed(add_attention))

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            if is_final_block is True: _upsample_type = None
            else: _upsample_type = upsample_type or "conv"  # default to 'conv'
            _norm_num_groups = reversed_norm_num_groups[i]
            _attention_num_heads = reversed_attention_num_heads[i]
            _double_attention = reversed_double_attention[i]
            _pre_attention = reversed_pre_attention[i]
            _freq_embedding_dim = reversed_freq_embedding_dim[i]
            _time_embedding_dim = reversed_time_embedding_dim[i]
            _add_attention = reversed_add_attention[i]

            up_block = SeparableAttnUpBlock2D(
                num_layers=layers_per_block + 1,
                in_channels=0,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=_norm_num_groups,
                attention_num_heads=_attention_num_heads,
                resnet_time_scale_shift="group",
                upsample_type=_upsample_type,
                separate_attn_dim=separate_attn_dim_up,
                double_attention=_double_attention,
                pre_attention=_pre_attention,
                conv_size=conv_size,
                freq_embedding_dim=_freq_embedding_dim,
                time_embedding_dim=_time_embedding_dim,
                add_attention=_add_attention,
                use_skip_samples=False,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups[-1] != 0:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups[-1], eps=norm_eps)
        else:
            self.conv_norm_out = nn.Identity()
        self.conv_act = get_activation(act_fn)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.scale_norm = ScaleNorm(2)

        self.gradient_checkpointing = False

    def forward(self, z, latent_embeds=None):
        sample = z
        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # middle
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), sample, latent_embeds, use_reentrant=False
            )
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block), sample, latent_embeds, use_reentrant=False
                )

        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        scaled_sample = torch.zeros_like(sample)

        scaled_sample[:, :2, :, :] = self.scale_norm(sample[:, :2, :, :].sigmoid())
        #scaled_sample[:, :2, :, :] = sample[:, :2, :, :].sigmoid()

        """
        phase_x_squared = sample[:, 2, :, :].square()
        phase_y_squared = sample[:, 3, :, :].square()
        beta = 2 / (1 + torch.exp(-phase_x_squared - phase_y_squared)) - 1
        norm = (phase_x_squared + phase_y_squared).sqrt().clip(min=1e-8) ** beta
        waveform = sample[:, 2:, :, :] / norm.unsqueeze(1)
        scaled_sample[:, 2:, :, :] = waveform# - waveform.mean(dim=3, keepdim=True)
        """

        waveform = sample[:, 2:, :, :].tanh()
        #scaled_sample[:, 2:, :, :] = sample[:, 2:, :, :].tanh()
        scaled_sample[:, 2:, :, :] = waveform #- waveform.mean(dim=3, keepdim=True)

        return scaled_sample

class AutoencoderKLDual(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        act_fn: str = "silu",
        block_out_channels: Tuple[int] = (128,256,512,512,),
        latent_channels: int = 4,
        sample_size: Tuple[int, int] = (64,2048,),
        scaling_factor: float = 0.18215,
        layers_per_block=2,
        layers_per_mid_block: int = 1,
        norm_num_groups: Union[int, Tuple[int]] = 32,
        norm_eps: float = 1e-6,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        add_mid_attention: bool = True,
        attention_num_heads: Union[int, Tuple[int]] = (8,16,32,32),
        separate_attn_dim_down: Tuple[int] = (3,3),
        separate_attn_dim_up: Tuple[int] = (3,3,3),    
        separate_attn_dim_mid: Tuple[int] = (0,),
        double_attention: Union[bool, Tuple[bool]] = False,
        pre_attention: Union[bool, Tuple[bool]] = False,
        conv_size = (3,3),
        freq_embedding_dim: Union[int, Tuple[int]] = 256,
        time_embedding_dim: Union[int, Tuple[int]] = 256,
        add_attention: Union[bool, Tuple[bool]] = True,
        multiscale_spectral_loss: dict = None,
        last_global_step: int = 0,
    ):
        super().__init__()

        if not isinstance(attention_num_heads, int) and len(attention_num_heads) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `attention_num_heads` as `block_out_channels`. `attention_num_heads`: {attention_num_heads}. `block_out_channels`: {block_out_channels}."
            )   
        
        if not isinstance(norm_num_groups, int) and len(norm_num_groups) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `norm_num_groups` as `block_out_channels`. `norm_num_groups`: {norm_num_groups}. `block_out_channels`: {block_out_channels}."
            ) 
        
        if not isinstance(double_attention, bool) and len(double_attention) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `double_attention` as `block_out_channels`. `double_attention`: {double_attention}. `block_out_channels`: {block_out_channels}."
            )
        
        if not isinstance(pre_attention, bool) and len(pre_attention) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `pre_attention` as `block_out_channels`. `pre_attention`: {pre_attention}. `block_out_channels`: {block_out_channels}."
            )
        
        if not isinstance(freq_embedding_dim, int) and len(freq_embedding_dim) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `freq_embedding_dim` as `block_out_channels`. `freq_embedding_dim`: {freq_embedding_dim}. `block_out_channels`: {block_out_channels}."
            ) 
        
        if not isinstance(time_embedding_dim, int) and len(time_embedding_dim) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `time_embedding_dim` as `block_out_channels`. `time_embedding_dim`: {time_embedding_dim}. `block_out_channels`: {block_out_channels}."
            ) 
        
        if not isinstance(add_attention, bool) and len(add_attention) != len(block_out_channels):
            raise ValueError(
                f"Must provide the same number of `add_attention` as `block_out_channels`. `add_attention`: {add_attention}. `block_out_channels`: {block_out_channels}."
            )
        
        if isinstance(attention_num_heads, int):
            attention_num_heads = (attention_num_heads,) * len(block_out_channels)
        if isinstance(norm_num_groups, int):
            norm_num_groups = (norm_num_groups,) * len(block_out_channels)
        if isinstance(separate_attn_dim_down, int):
            separate_attn_dim_down = (separate_attn_dim_down,)
        if isinstance(separate_attn_dim_up, int):
            separate_attn_dim_up = (separate_attn_dim_up,)
        if isinstance(double_attention, bool):
            double_attention = (double_attention,) * len(block_out_channels)
        if isinstance(pre_attention, bool):
            pre_attention = (pre_attention,) * len(block_out_channels)
        if isinstance(freq_embedding_dim, int):
            freq_embedding_dim = (freq_embedding_dim,) * len(block_out_channels)
        if isinstance(time_embedding_dim, int):
            time_embedding_dim = (time_embedding_dim,) * len(block_out_channels)
        if isinstance(add_attention, bool):
            add_attention = (add_attention,) * len(block_out_channels)

        # pass init params to Encoder
        self.encoder = EncoderDual(
            in_channels=in_channels,
            out_channels=latent_channels,
            act_fn=act_fn,
            double_z=True,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            layers_per_mid_block=layers_per_mid_block,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            downsample_type=downsample_type,
            add_mid_attention=add_mid_attention,
            attention_num_heads=attention_num_heads,
            separate_attn_dim_down=separate_attn_dim_down,
            separate_attn_dim_mid=separate_attn_dim_mid,
            double_attention=double_attention,
            pre_attention=pre_attention,
            conv_size = conv_size,
            freq_embedding_dim=freq_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            add_attention=add_attention,
        )

        # pass init params to Decoder
        self.decoder = DecoderDual(
            in_channels=latent_channels,
            out_channels=out_channels,
            act_fn=act_fn,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            layers_per_mid_block=layers_per_mid_block,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            upsample_type=upsample_type,
            add_mid_attention=add_mid_attention,
            attention_num_heads=attention_num_heads,
            separate_attn_dim_up=separate_attn_dim_up,
            separate_attn_dim_mid=separate_attn_dim_mid,
            double_attention=double_attention,
            pre_attention=pre_attention,
            conv_size = conv_size,
            freq_embedding_dim=freq_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            add_attention=add_attention,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

        if multiscale_spectral_loss is not None:
            mss_version = multiscale_spectral_loss.get("version", 1)
            if mss_version == 1:
                self.multiscale_spectral_loss = DualMultiscaleSpectralLoss(multiscale_spectral_loss)
            elif mss_version == 2:
                self.multiscale_spectral_loss = DualMultiscaleSpectralLoss2(multiscale_spectral_loss)
            elif mss_version == 3:
                self.multiscale_spectral_loss = DualMultiscaleSpectralLoss3(multiscale_spectral_loss)
            elif mss_version == 4:
                self.multiscale_spectral_loss = DualMultiscaleSpectralLoss4(multiscale_spectral_loss)
            else:
                raise ValueError("Invalid multiscale_spectral_loss version")
        else:
            raise ValueError("Must provide multiscale_spectral_loss_params")

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (EncoderDual, DecoderDual)):
            module.gradient_checkpointing = value

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
