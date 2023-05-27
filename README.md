# Fourier Dual Diffusion

This is an experimental diffusers pipeline for audio (music).

- The pipeline uses 2 unets, one for the time domain and one for the frequency domain.
- The 2 unets are processed one after the other in each diffusion step with a fourier transform in between.
- Although a sample can be very large (47 seconds at 44100hz is 208896 samples), each unet is trained on and processes the signal in windowed overlapping chunks to conserve memory.
- The fourier transform is done on the _whole_ signal after the corresponding unet does a denoising step, and guarantees that information in any _one_ will be visible to _all_ chunks in just 2 diffusion steps.
- Instead of gaussian white noise, each unet is trained with fractal (1/f or "pink") noise, which is intended to encourage the unets to learn multi-scale patterns. 
- The idea is based on n log n complexity of the attention mechanism introduced in [Hyena Hierarchy](https://arxiv.org/abs/2302.10866), but applied in the context of a generative diffusion model.
