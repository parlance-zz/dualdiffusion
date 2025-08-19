# Dual Diffusion
Dual Diffusion is a generative diffusion model for music. This model and the code in this repository is still a work-in-progress.

The current model uses a custom architecture based on the [EDM2](https://github.com/NVlabs/edm2) UNet and a custom 2-stage VAE design with a diffusion decoder. The model uses [CLAP](https://github.com/LAION-AI/CLAP) audio embeddings for conditioning and is trained on video game music from the mid-90s to present day.

You can hear demo audio from the model in various stages of training / development [here](https://www.g-diffuser.com/dualdiffusion).

I started this project in August/2023 with the intention of achieving 3 goals:
* Familiarize myself with every component of modern diffusion models in both inference and training
* Train a model from scratch that is able to generate music that I would actually want to listen to
* Do the above using only desktop GPU hardware

The model has changed substantially over the course of development in the last 18 months.

* Initially (August/2023) the diffusion model was unconditional and worked directly on raw audio.
   * Due to memory and performance constraints this meant I was limited to ~16 seconds of mono audio at 8khz.
   * I experimented with 1d and 2d formats with various preprocessing steps. I found 2d formats were able to generalize better with a small dataset and were more compute and parameter efficient than 1d formats.
   * For 2d formats I found using separable attention (merging the rows / columns with the batch dimension alternately) could make using attention in a high dimensionality model practical without sacrificing too much quality.
   * I found that attention was an absolute requirement to allow the model to understand the perceptual ~log-frequency scale in music when the format uses a linear-frequency scale, especially with positional embeddings for the frequency axis.
   * I found that v-prediction with a cosine-based schedule worked significantly better than any alternatives (for raw audio / non-latent diffusion).

* In December/2023 I began training variational auto-encoder models to try a latent diffusion model.
   * After moving to latent diffusion I was able to begin training the diffusion model on crops of 45 seconds @ 32khz stereo
   * I found that point-wise loss works very poorly in terms of reconstruction quality, and multi-scale spectral loss was a much better option. I found that multi-scale spectral power density loss works considerably better for music and learning pitch if you add an appropriately weighted loss term for phase.
   * I found that although it was possible for a VAE that includes phase to have good reconstruction quality and a compact latent space, the latent space had high entropy and was not easily interpretable by the latent diffusion model. This resulted in in a model that could generate good quality _sounds_ but had poor musicality.
   * I found that the latent diffusion model performs substantially better with input latents that have uniform variance; there is no need for the log-variance to be a learnable parameter, instead I predefine a uniform target SNR for the latent distribution.
   * I found that the latent diffusion model performance is substantially improved with lower target snr / higher log-variance in the latent distribution, provided the latent distribution _mode_ is used for the training target. Lower target snr allows the latent diffusion model to exclude low noise levels during both training and inference; the residual noise after sampling can be designed to match the expected variance in the latent distribution.

* In March/2024 I started using mel-scale spectrogram based formats, excluding phase information.
   * I found that it was possible to considerably improve the FGLA phase reconstruction by tuning the window and spectrogram parameters, as well as modifying the algorithm to anneal phases for stereo signals in a more coherent way. I settled on a set of parameters that resulted in a spectrogram dimensionality that is the same as the critically sampled raw audio without sacrificing too much perceptual quality.
   * I found that multi-scale spectral loss works well 2d for spectrogram / image data, the resulting quality is somewhere between point-wise loss and adversarial loss.

* In April/2024 I replaced the diffusion model and VAE unets (previously based on the unconditional unet in [diffusers](https://github.com/huggingface/diffusers)) with the improved [edm2](https://github.com/NVlabs/edm2) unet.
   * I made several improvements to the edm2 unet:
      * Replaced einsum attention with torch sdp attention
      * Replaced fourier embedding frequencies / phases for a smoother inner product space
      * Added class label dropout in a way that preserves magnitude on expectation
      * Replaced the weight normalization in the forward method of the mpconv module with weight normalization that is only applied when the weights are updated for improved performance and lower memory consumption
      * Added a correction when using dropout inside blocks to preserve magnitude on expectation during inference
      * Replaced the up/downsample with equivalent torch built-ins for improved performance
      * Merged some of the pre-conditioning code into the unet itself.
   * I started using torch dynamo / model compilation and added the appropriate compiler hints for maximum performance.
   * I started using class label-based conditioning and implemented classifier free guidance for a major improvement in quality and control.

* In May/2024 I adopted the edm/ddim noise schedule, sampling algorithm, and learn rate schedule.
   * I found that the log-normal sigma sampling in training could be improved by using the per-sigma estimated error variance to calculate a sigma sampling pdf that explicitly focuses on the noise levels that the model is most capable of making progress on.
   * I found that using stratified sampling to distribute sigmas as evenly as possible within each mini-batch could mitigate problems with smaller mini-batches.
   * I began pre-encoding the latents for my dataset before diffusion model training for increased performance and reduced memory consumption. I found pre-encoding the latents _before_ random crop can negatively influence generated sample quality due to the lack of variations created by sub-latent-pixel offsets. I added pre-encoded latent variations for those offsets.
   * I began training with EMA weights of multiple lengths.

* In June-July/2024 I experimented with the model architecture and added noise in sampling.
   * I found that using low-rank linear layers in the model resnet blocks could significantly increase quality without drastically increasing the number of parameters; Specifically, projecting to a higher number of dimensions inside the resnet block where the non-linearity is applied and then projecting back down.
   * I found that adding conditioned modulation to the self-attention qkv linear layers significantly increased conditioning quality.
   * I trained another 1d model using the all the new architecture improvements and confirmed my earlier findings that 1d models generalize poorly with a small dataset.
   * Inspired by [this paper](https://arxiv.org/abs/2310.17467v2) I experimented with adding a large amount of noise to prolong the number of steps where the model is in a critical or near critical state during sampling. This significantly increased the quality of the samples while drastically lowering the resulting sample temperature which is desirable for music.

* In August/2024 I began a near-complete rewrite of the codebase to pay down the technical debt that had accumulated over the last year of rapid experimentation.
   * I re-implemented the last of the remaining code that used [diffusers](https://github.com/huggingface/diffusers) and removed the dependency.
   * I built a modular system to allow for easy experimentation with the training process for arbitrary modules.
   * The training code was re-writen and uses configuration files as much as possible to allow for automated hyperparameter or model architecture search.
   * Dataset pre-processing code was completely re-written. I intend to spend more time developing tools / models to filter and augment the dataset which will allow me to use a larger volume of low quality data.

* In September/2024 I began training inpainting models and started building a webUI.
   * I found that the inpainting model conversion / intialization strategy (zero-init extra channels for reference sample and mask in the first conv_in layer) used by stable diffusion works with my model but can take a while to properly converge. Adding inpainting to the existing model reduced non-inpainting performance slightly so I'm using the inpainting model exclusively for inpainting tasks for now. I'd like to find a way to improve the performance to the point where I only need one model for both tasks.
   * For the webUI I started with Gradio but found unfortunate limitations and poor performance. Looking for alternatives I found [NiceGUI](https://github.com/zauberzeug/nicegui) as a much better fit for what I want to accomplish. As of today the webUI is functional but is missing webUI controlled model switching and some of the detailed debug information / plots that I was using for development. The default HTML5 audio element is clunky and I'll be working to replace it with a custom player more suitable for audio editing.

* In October/2024 I continued developing the webUI and experimenting with EMA techniques.
   * I replaced the default HTML5 audio elements with a custom niceGUI element that shows a spectrogram view and
   includes a precise time cursor. This improved the ease of use and precision of the in-painting tool.
   * I added out-painting to extend or prepend generated samples. I also added an option to generate seamless loops.
   * I separated the PyTorch model processing into its own process to make the UI more responsive.
   * I found [SwitchEMA](https://github.com/Westlake-AI/SEMA) could make a significant improvement in validation loss with no additional cost / training overhead.
   * After some experimentation I found that the technique could be further improved by instead using feedback from EMA weights back into the train weights with a hyperparameter to control the feedback strength.

* In November/2024 I began the process of adopting [CLAP](https://github.com/LAION-AI/CLAP) to replace the class label conditioning system with the goal of training a model with a cleaner expanded dataset.
   * I replaced some of the dataset preparation workflow by using [foobar2000](https://www.foobar2000.org/) for transcoding and labelling.
   * I added pre-encoded CLAP embeddings to the existing pre-encoded latents for samples in the dataset. The scores for these embeddings against a pre-defined list of labels / captions and negative examples are used to update the audio metadata and bootstrap the process of cleaning / filtering the dataset.
   * I began experimenting with different ways to integrate CLAP conditioning into the model training. I found that using the aggregate average audio embedding for the entire dataset can be used effectively as the unconditional embedding when using conditioning dropout and classifier-free guidance. I found the aggregate audio and text embeddings for what use to be a simple class label can be used effectively to sample from that class.

* In December/2024 I began the process of rewriting most of the dataset pre/post-processing code and assembling a new dataset with modern video game music with fully orchestrated or recorded scores.
   * I wanted the new dataset to be substantially larger so much effort was spent on building a dataset processing pipeline that is as efficient as possible.
   * For this new dataset I wanted the model to better understand natural dynamics / loudness variation so I spent quite a bit of time making the dataset pre-processing / normalization more robust to avoid
   the need for forced normalization anywhere further down the line in the model pipeline.

* In January/2025 I started working on new VAE architectures to better suit the qualities of the new dataset with realistic audio and improved dynamics.
   * In December/2023 I noted that certain VAE design and training choices can result in high quality audio, but at the cost of the latent diffusion model performance owing to chaotic latents that are difficult to interpret. For the new VAE I wanted to maximize latent interpretability as much as possible, even at the expense of audio quality.
   * I found that training a VAE with plain MSE loss results in very interpretable latents (albeit only if the encoder training is frozen after a modest number of training steps). The latent diffusion model performs significantly better when using these latents as all the information about the fine details has been removed. The audio quality is poor as expected.
   * I found that it is possible to remedy the audio quality problem by training a secondary diffusion model to act as the 2nd stage in a 2-stage VAE decoding process: The 1st stage VAE encodes and decodes the latents
   to a blurry audio spectrogram - at this point all fine detail has been lost. The 2nd stage is a standard diffusion model that uses the blurry audio spectrogram as conditioning in a similar way to most inpainting models (although in this case nothing is masked).
   * The diffusion model decoder is extremely good at accurately reconstructing these highly localized fine details and the resulting audio quality far exceeds what I was able to achieve with any single stage VAE and complex loss functions.

* In March/2025 I continued to experiment with VAE and diffusion decoder designs to improve audio quality.
   * Instead of using a diffusion decoder to improve the mel-scale spectrogram for FGLA I found it is possible to use the mel-spec as conditioning for a diffusion decoder that operates directly on an MDCT of the raw audio. Because the mel-scale and linear-scale frequency resolutions are different, the mel-scale spectrogram is first upsampled to a high resolution linear-frequency scale spectrogram, and then subsequently chunked into channels to match the frequency resolution of the MDCT.
   * The performance of the MDCT diffusion decoder can be further improved by reducing the range of noise scales in the noise schedule. This can be done without sacrificing linearity by scaling MDCT coefficients by a factor inversely proportional to their wavelength. This rescaling is then inverted when the MDCT is decoded to recover the original frequency response.
   * To further improve the interpretability of the latents I starting experimenting with an encoder that operates without downsampling. Only after projecting to the final latent channel count are the latents downsampled (using average pooling) to the desired latent resolution. This effectively "supersamples" the latents and guarantees sub-latent-pixel shift equivariance without the need for any complex augmentations (as in EQ-VAE) or expensive filtering operations in the encoder (as in StyleGAN3).
   * The supersampled latent encoder works extremely well with *2D* multi-scale power-spectral-density loss with appropriate weighting for each frequency (inversely proportional to its wavelength). Prime-sized block widths with randomized offsets and a flat-top window are used for best results.

* In July/2025 I began training a larger model using all the above improvements (U3).


Some additional notes:
* The web UI is currently broken because of the move from class label to CLAP conditioning back in November/2024. I'll fix it at some point but my top priority is developing the model itself. Batch sampling is still available from a JSON config file.
* Some areas of the codebase need to be cleaned up and refactored / de-duplicated. Due to the experimental nature of the project the need for rapid iteration outweighs my desire for perfectly clean code.
* SNES game music was used as my dataset for the first year of development (2023). The SNES dataset was comprised of ~20,000 samples of lengths between 1 and 3 minutes. I used the each game as a class label for conditioning, which means you could choose a game (or a weighted combination of multiple games) to generate new music with the appropriate instruments and style. The number of examples per class / game was anywhere from ~5 to ~50 and all generated samples combining more than 1 game were "zero-shot".
* The training code supports multiple GPUs and distributed training through huggingface accelerate, currently logging is to tensorboard.
* The dataset pre/post-processing code is included in this repository which includes everything needed to train a new model on your own data
* All the code in this repository is tested to work on both Windows and Linux platforms, although performance is significantly better on Linux
