import torch
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from .modified_unet_2d_condition import Modified_UNet2DConditionModel

def UNET():
    # default init parameter doesn't totally match the sd-v1.5 unet block
    # so we have to manually specify it
    unet = UNet2DConditionModel(
        sample_size=64,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=768,
    )

    parameter_path = "/data/yhc/stable-diffusion-one-flow/unet.pt"

    unet.load_state_dict(torch.load(parameter_path, map_location='cpu'))
    return unet

def Modified_UNet():
    # default init parameter doesn't totally match the sd-v1.5 unet block
    # so we have to manually specify it
    unet = Modified_UNet2DConditionModel(
        sample_size=64,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=768,
    )

    parameter_path = "/data/yhc/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin"

    unet.load_state_dict(torch.load(parameter_path, map_location='cpu'))
    return unet

