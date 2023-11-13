import torch
from diffusers.models.autoencoder_kl import AutoencoderKL

def VAE():
    # default init parameter doesn't totally match the sd-v1.5 vae block
    # so we have to manually specify it
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        act_fn='silu',
        latent_channels=4,
        norm_num_groups=32,
        scaling_factor=0.18215
    )

    parameter_path = "/data/yhc/stable-diffusion-one-flow/vae.pt"

    vae.load_state_dict(torch.load(parameter_path, map_location='cpu'))
    return vae