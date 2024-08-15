
"""
            latent_shape = module.get_latent_shape(sample_shape)
            channel_kl_loss_weight = model_params["vae_training_params"]["channel_kl_loss_weight"]
            recon_loss_weight = model_params["vae_training_params"]["recon_loss_weight"]
            point_loss_weight = model_params["vae_training_params"]["point_loss_weight"]
            imag_loss_weight = model_params["vae_training_params"]["imag_loss_weight"]

            logger.info("Training VAE model:")
            logger.info(f"VAE Training params: {dict_str(model_params['vae_training_params'])}")
            logger.info(f"Channel KL loss weight: {channel_kl_loss_weight}")
            logger.info(f"Recon loss weight: {recon_loss_weight} - Point loss weight: {point_loss_weight} - Imag loss weight: {imag_loss_weight}")

            target_snr = module.get_target_snr()
            target_noise_std = (1 / (target_snr**2 + 1))**0.5
            logger.info(f"VAE Target SNR: {target_snr:{8}f}")

            channel_kl_loss_weight = torch.tensor(channel_kl_loss_weight, device=accelerator.device, dtype=torch.float32)
            recon_loss_weight = torch.tensor(recon_loss_weight, device=accelerator.device, dtype=torch.float32)
            point_loss_weight = torch.tensor(point_loss_weight, device=accelerator.device, dtype=torch.float32)
            imag_loss_weight = torch.tensor(imag_loss_weight, device=accelerator.device, dtype=torch.float32)

            module_log_channels = [
                "channel_kl_loss_weight",
                "recon_loss_weight",
                "imag_loss_weight",
                "real_loss",
                "imag_loss",
                "channel_kl_loss",
                "latents_mean",
                "latents_std",
                "latents_snr",
                "point_similarity_loss",
                "point_loss_weight",
            ]
"""


"""

                    raw_samples = batch["input"]
                    sample_game_ids = batch["game_ids"]
                    sample_t_ranges = batch["t_ranges"] if self.dataset.config.t_scale is not None else None
                    raw_sample_paths = batch["sample_paths"]
                    #sample_author_ids = batch["author_ids"]
                    
                        samples_dict = pipeline.format.raw_to_sample(raw_samples, return_dict=True)
                        vae_class_embeddings = module.get_class_embeddings(pipeline.get_class_labels(sample_game_ids))
                        
                        posterior = module.encode(samples_dict["samples"],
                                                vae_class_embeddings,
                                                pipeline.format)
                        latents = posterior.sample(pipeline.noise_fn)
                        latents_mean = latents.mean()
                        latents_std = latents.std()

                        measured_sample_std = (latents_std**2 - target_noise_std**2).clip(min=0)**0.5
                        latents_snr = measured_sample_std / target_noise_std
                        model_output = module.decode(latents,
                                                    vae_class_embeddings,
                                                    pipeline.format)

                        recon_samples_dict = pipeline.format.sample_to_raw(model_output, return_dict=True, decode=False)
                        point_similarity_loss = (samples_dict["samples"] - recon_samples_dict["samples"]).abs().mean()
                        
                        recon_loss_logvar = module.get_recon_loss_logvar()
                        real_loss, imag_loss = pipeline.format.get_loss(recon_samples_dict, samples_dict)
                        real_nll_loss = (real_loss / recon_loss_logvar.exp() + recon_loss_logvar) * recon_loss_weight
                        imag_nll_loss = (imag_loss / recon_loss_logvar.exp() + recon_loss_logvar) * (recon_loss_weight * imag_loss_weight)

                        latents_square_norm = (torch.linalg.vector_norm(latents, dim=(1,2,3), dtype=torch.float32) / latents[0].numel()**0.5).square()
                        latents_batch_mean = latents.mean(dim=(1,2,3))
                        channel_kl_loss = (latents_batch_mean.square() + latents_square_norm - 1 - latents_square_norm.log()).mean()
                        
                        loss = real_nll_loss + imag_nll_loss + channel_kl_loss * channel_kl_loss_weight + point_similarity_loss * point_loss_weight
"""