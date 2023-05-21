# Fourier Dual Diffusion

This is an experimental diffusers pipeline for audio.

- The pipeline uses 2 unets, one for the time domain and one for the frequency domain.
- The 2 unets are processed one after the other in each diffusion step with a fourier transform in between.
- Although a sample can be very large (47 seconds at 44100hz is 208896 samples), each unet is trained on and processes the signal in windowed overlapping chunks to conserve memory.
- The fourier transform is done on the whole signal after the corresponding unet does a denoising step on the signal, and guarantees that information in any one chunk can reach any other chunk in only 2 diffusion steps.
- The idea is based on n log n complexity of the attention mechanism introduced in [Hyena Hierarchy]https://arxiv.org/abs/2302.10866, but applied in the context of a generative diffusion model.
