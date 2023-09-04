from diffusers.models import UNet1DModel

model = UNet1DModel(
            sample_size = 256,
            sample_rate = 8000,
            in_channels = 256,
            out_channels = 256,
            extra_in_channels = 0,
            time_embedding_type = "positional",
            flip_sin_to_cos = True,
            use_timestep_embedding = True,
            freq_shift = 0.0,
            down_block_types = ("DownBlock1D",
                                "DownBlock1D",
                                "AttnDownBlock1D",
                                "AttnDownBlock1D",
                                "AttnDownBlock1D",
                                "AttnDownBlock1D",),
            up_block_types = ("AttnUpBlock1D",
                              "AttnUpBlock1D",
                              "AttnUpBlock1D",
                              "AttnUpBlock1D",
                              "UpBlock1D",
                              "UpBlock1D",),
            mid_block_type = "UNetMidBlock1D",
            out_block_type = None,
            block_out_channels = (512, 768, 1024, 1280, 1536, 1792),
            act_fn = "silu",
            norm_num_groups = 32,
            layers_per_block = 2,
            downsample_each_block = False,
)

model.save_pretrained("./models/unet1d")