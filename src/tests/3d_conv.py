import argparse
import time

import torch

from modules.mp_tools import normalize

from modules.daes.dae_edm2_q432s import DAE, DAE_Config
from modules.mp_stereo_conv import MPStereoConv


def stereo_conv3d_reference(x: torch.Tensor, conv: MPStereoConv, gain: float = 1.) -> torch.Tensor:

    w = conv.weight.float()
    w = normalize(w) if conv.disable_weight_norm == False else w
    w = w * (gain / w[0].numel()**0.5)

    B, _, H, W = x.shape
    x = x.view(B, -1, 2, H, W)
    x = torch.cat((x, x[:, :, 0:1]), dim=2) # one-sided reflect padding in z
    x = torch.nn.functional.conv3d(x, w.to(x.dtype), padding=(0, *conv.padding), groups=conv.groups, stride=(1, *torch.nn.modules.utils._pair(conv.stride)))

    return x.reshape(B, -1, x.shape[3], x.shape[4])

@torch.no_grad()
def check_correctness(device: torch.device) -> None:

    for kernel, groups, stride in [((3,3), 1, 1), ((3,3), 2, 1), ((1,1), 1, 1), ((3,3), 4, 2)]:
        x = torch.randn(3, 2 * 16, 12, 10, device=device)
        conv = MPStereoConv(16, 32, kernel, groups=groups, stride=stride).to(device).train()

        y_ref = stereo_conv3d_reference(x, conv, gain=0.7)
        y = conv(x, gain=0.7)

        assert y.shape == y_ref.shape, f"output shape {y.shape} does not match conv3d reference {y_ref.shape}"
        error = ((y - y_ref).abs().max() / y_ref.abs().max()).item()
        print(f"correctness kernel={kernel} groups={groups} stride={stride} rel_err={error:.2e}")
        assert error < 5e-3, "MPStereoConv does not match conv3d reference"

def benchmark(args: argparse.Namespace, device: torch.device) -> None:

    torch.manual_seed(0)
    torch._dynamo.reset()

    config = DAE_Config(
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        latent_channels=args.latent_channels,
        in_num_freqs=args.num_freqs,
        in_psd_freqs=args.num_freqs * 2,
        model_channels=args.model_channels,
    )

    model = DAE(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    compiled_model = torch.compile(model, fullgraph=True, dynamic=False)

    samples = torch.randn(args.batch_size, config.in_channels * 2, config.in_psd_freqs, args.length, device=device)

    def train_step() -> torch.Tensor:
        _, recon, _, _, _ = compiled_model(samples, None)
        loss = torch.nn.functional.mse_loss(recon.float(), samples)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss

    start_time = time.perf_counter()
    for _ in range(args.warmup_steps):
        train_step()
    torch.cuda.synchronize()
    compile_time = time.perf_counter() - start_time

    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(args.steps):
        loss = train_step()
    end_event.record()
    torch.cuda.synchronize()

    ms_per_step = start_event.elapsed_time(end_event) / args.steps
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{ms_per_step:8.2f} ms/step  peak mem {peak_mem_gb:6.2f} GB  "
          f"params {num_params/1e6:.2f}M  warmup+compile {compile_time:6.1f} s  final loss {loss.item():.4f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--num_freqs", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--model_channels", type=int, default=96)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--skip_check", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    #if not args.skip_check:
    #    check_correctness(device)

    benchmark(args, device)
