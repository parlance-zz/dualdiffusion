import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from unet_edm2 import Block, MPConv, mp_silu, normalize

class IsotropicGaussianDistribution(object):

    def __init__(self, parameters, logvar, deterministic=False):

        self.deterministic = deterministic
        self.parameters = self.mean = parameters
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )
        else:
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)

    def sample(self, noise_fn, **kwargs): 
        noise = noise_fn(self.mean.shape,
                         device=self.parameters.device,
                         dtype=self.parameters.dtype,
                         **kwargs)
        return self.mean + self.std * noise

    def mode(self):
        return self.mean
    
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0], device=self.parameters.device, dtype=self.parameters.dtype)
        
        reduction_dims = tuple(range(0, len(self.mean.shape)))
        if other is None:
            return 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=reduction_dims)
        else:
            return 0.5 * torch.mean(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=reduction_dims,
            )
    
class AutoencoderKL_EDM2(ModelMixin, ConfigMixin):

    __constants__ = ["label_dim", "dropout", "target_snr", "num_blocks"]

    @register_to_config
    def __init__(self,
        in_channels = 2,                    # Number of input channels.
        out_channels = 2,                   # Number of output channels.
        latent_channels = 4,                # Number of channels in latent space.
        target_snr = 2.,                    # The learned latent snr will not exceed this snr
        channels_per_head = 64,             # Number of channels per attention head.
        label_dim = 0,                      # Class label dimensionality. 0 = unconditional.
        dropout = 0,                        # Dropout probability.
        model_channels       = 64,          # Base multiplier for the number of channels.
        channel_mult         = [1,2,3,5],   # Per-resolution multipliers for the number of channels.
        channel_mult_emb     = None,        # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_layers_per_block = 3,           # Number of residual blocks per resolution.
        attn_levels          = [],          # List of resolutions with self-attention.
        midblock_decoder_attn = False,      # Whether to use attention in the first layer of the decoder.
        #**block_kwargs,                    # Arguments for Block.
        last_global_step = 0,               # Only used to track training progress in config.
    ):
        super().__init__()

        block_kwargs = {"channels_per_head": channels_per_head, "dropout": dropout}

        cblock = [int(model_channels * x) for x in channel_mult]
        if label_dim != 0:
            cemb = int(model_channels * channel_mult_emb) if channel_mult_emb is not None else max(cblock)
        else:
            cemb = 0

        self.label_dim = label_dim
        self.dropout = dropout
        self.target_snr = target_snr
        self.num_blocks = len(channel_mult)

        target_noise_std = (1 / (self.target_snr**2 + 1))**0.5
        target_sample_std = (1 - target_noise_std**2)**0.5
        self.latents_out_gain = torch.nn.Parameter(torch.tensor(target_sample_std))
        self.out_gain = torch.nn.Parameter(torch.ones([]))
        
        # Embedding.
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Training uncertainty estimation.
        self.recon_loss_logvar = torch.nn.Parameter(torch.zeros(1))
        self.latents_logvar = torch.nn.Parameter(torch.zeros(1))
        
        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels + 2 # 1 extra const channel, 1 pos embedding channel
        for level, channels in enumerate(cblock):
            
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'conv_in'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'block{level}_down'] = Block(cout, cout, cemb, 0,
                                                       flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_layers_per_block):
                cin = cout
                cout = channels
                self.enc[f'block{level}_layer{idx}'] = Block(cin, cout, cemb, 0,
                                                             flavor='enc', attention=(level in attn_levels), **block_kwargs)

        self.conv_latents_out = MPConv(cout, latent_channels, kernel=[3,3])
        self.conv_latents_in = MPConv(latent_channels + 2, cout, kernel=[3,3]) # 1 extra const channel, 1 pos embedding channel

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, channels in reversed(list(enumerate(cblock))):
            
            if level == len(cblock) - 1:
                self.dec[f'block{level}_in0'] = Block(cout, cout, cemb, 0,
                                                      flavor='dec',attention=midblock_decoder_attn, **block_kwargs)
                self.dec[f'block{level}_in1'] = Block(cout, cout, cemb, 0,
                                                      flavor='dec', **block_kwargs)
            else:
                self.dec[f'block{level}_up'] = Block(cout, cout, cemb, 0,
                                                     flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_layers_per_block + 1):
                cin = cout
                cout = channels
                self.dec[f'block{level}_layer{idx}'] = Block(cin, cout, cemb, 0,
                                                             flavor='dec', attention=(level in attn_levels), **block_kwargs)
        self.conv_out = MPConv(cout, out_channels, kernel=[3,3])

    def get_class_embeddings(self, class_labels):
        return mp_silu(self.emb_label(normalize(class_labels).to(device=self.device, dtype=self.dtype)))

    @torch.compile(fullgraph=True)
    def encode(self, x, class_embeddings, format):

        x = torch.cat((x, torch.ones_like(x[:, :1]),
                       format.get_positional_embedding(x, None, mode="linear")), dim=1)
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, class_embeddings, None, None)

        latents = self.conv_latents_out(x, gain=self.latents_out_gain)
        noise_logvar = torch.tensor(np.log(1 / (self.target_snr**2 + 1)), device=x.device, dtype=x.dtype)
        return IsotropicGaussianDistribution(latents, noise_logvar)
    
    @torch.compile(fullgraph=True)
    def decode(self, x, class_embeddings, format):
        
        x = torch.cat((x, torch.ones_like(x[:, :1]),
                       format.get_positional_embedding(x, None, mode="linear")), dim=1)
        x = self.conv_latents_in(x)
        for _, block in self.dec.items():
            x = block(x, class_embeddings, None, None)

        return self.conv_out(x, gain=self.out_gain)
    
    def get_recon_loss_logvar(self):
        return self.recon_loss_logvar
    
    def get_target_snr(self):
        return self.target_snr
    
    def get_latent_shape(self, sample_shape):
        if len(sample_shape) == 4:
            return (sample_shape[0],
                    self.config.latent_channels,
                    sample_shape[2] // 2 ** (self.num_blocks-1),
                    sample_shape[3] // 2 ** (self.num_blocks-1))
        else:
            raise ValueError(f"Invalid sample shape: {sample_shape}")
    
    @torch.no_grad()
    def normalize_weights(self):
        for module in self.modules():
            if isinstance(module, MPConv):
                module.normalize_weights()