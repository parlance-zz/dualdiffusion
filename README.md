# Dual Diffusion
Dual Diffusion is a generative diffusion model for music. This model and the code in this repository is still a work-in-progress.

I'm currently using music from SNES games as my dataset during development. The dataset is comprised of 21,500 samples of lengths between 1 and 3 minutes. I am using the game the music is from as a class label for conditioning, which means you can choose a game (or a weighted combination of multiple games) to generate new music with the appropriate instruments and style. You can hear some samples generated by the model in various stages of training / development [here](https://drive.google.com/drive/folders/1aCs0HWvocO1-EN2dXhEa3Hwcje1xeO24?usp=drive_link).

I started this project in August/2023 with the intention of achieving 2 goals:
* Familiarize myself with every component of modern diffusion models in both inference and training
* Train a model from scratch that is able to generate music that I would actually want to listen to
* Do the above using only a single consumer GPU (4090)

The model has changed substantially over the course of development.
* Initially (August/2023) the diffusion model worked directly on raw audio.
   * Due to memory and performance constraints this meant I was limited to ~16 seconds of mono audio at 8khz.
   * I experimented with 1d and 2d formats with various preprocessing steps. I found 2d formats were able to generalize better with a small dataset and were more compute and parameter efficient than 1d formats.
   * For 2d formats I also found using separable attention (merging the rows / columns with the batch dimension alternately) could make using attention in a high dimensionality model practical without sacrificing too much quality.
   * I found that attention was an absolute requirement to allow the model to understand the perceptual ~log-frequency scale in music when the format uses a linear-frequency scale, especially with positional embeddings for the frequency axis.
   * I also found that v-prediction with a cosine-based schedule worked significantly better than any alternatives.
* In December/2023 I began training variational auto-encoder models to try a latent diffusion model.
   * After moving to latent diffusion I was able to begin training the diffusion model on crops of 45 seconds @ 32khz stereo
   * I found that point-wise loss works very poorly in terms of reconstruction quality, and multi-scale spectral loss was a much better option. I found that multi-scale spectral power density loss works considerably better for music and learning pitch if you add an appropriately weighted loss term for phase.
   * I found that although it was possible to train a VAE including the signal phase to have good reconstruction quality and a compact latent space, the latent space was not easily interpretable by the latent diffusion model.
   * I also found that the latent diffusion model performs substantially better with isotropic gaussian latents (rather than just diagonal gaussian); there is also no need for the log-variance to be a learnable parameter, instead I pre-define a target snr for the latent distribution.

* In March/2024 I started using mel-scale spectrogram based formats, excluding phase information.
   * I found that it was possible to considerably improve the FGLA phase reconstruction by tuning the window and spectrogram parameters, as well as modifying the algorithm to anneal phases for stereo signals in a more coherent way. I settled on a set of parameters that resulted in a spectrogram dimensionality that is the same as the critically sampled raw audio without sacrificing too much perceptual quality.
   * I found that multi-scale spectral loss works well 2d for spectrogram / image data, the resulting quality is somewhere between point-wise loss and adversarial loss.

* In April/2024 I replaced the diffusion model unet (based on the existing unconditional unet in [diffusers](https://github.com/huggingface/diffusers)) with the improved [edm2](https://github.com/NVlabs/edm2) unet.
   * I made several improvements to the edm2 unet:
      * Replaced einsum attention with torch sdp attention
      * Replaced fourier embedding frequencies / phases for a smoother inner product space
      * Added class label dropout in a way that preserves magnitude on expectation
      * Replaced the weight normalization in the forward method of the mpconv module with weight normalization that is only applied when the weights are updated for improved performance and lower memory consumption
      * Added correction for the output magnitude when using dropout inside blocks
      * Replaced the up/downsample with equivilent torch built-ins for improved performance
      * Merged some of the pre-conditioning code into the unet itself.
   * I started using torch dynamo / model compilation and added the appropriate compiler hints for maximum performance.
   * I also started using class label-based conditioning and implemented classifier free guidance for a major improvement in quality and control.

* In May/2024 I adopted the edm/ddim noise schedule, sampling algorithm, and learn rate schedule.
   * I found that the log-normal sigma sampling in training could be improved by using the per-sigma estimated error variance to calculate a sigma sampling pdf that explicitly focuses on the noise levels that the model is most capable of making progress on.
   * I found that using stratified sampling to distribute sigmas as evenly as possible within each mini-batch could mitigate problems with smaller mini-batches.
   * I began pre-encoding the latents for my dataset before diffusion model training for increased performance and reduced memory consumption. I found pre-encoding the latents before random crop can negatively influence model quality due to lacking the variations created by sub-latent-pixel offsets, I added pre-encoded latent variations for those offsets.
   * I also began training with EMA weights of multiple lengths.

Some additional notes:
* The training code supports multiple GPUs and distributed training through huggingface accelerate, currently logging is to tensorboard.
* The dataset pre/post-processing code is included in this repository which includes everything needed to train a new model on your own data
