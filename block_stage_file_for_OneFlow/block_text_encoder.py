import torch
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextConfig

def TEXT_ENCODER():
    CLIPTextConfig_path = "/data/yhc/stable-diffusion-v1-5/text_encoder"
    text_encoder = CLIPTextModel(CLIPTextConfig.from_pretrained(CLIPTextConfig_path))
    parameter_path = "/data/yhc/stable-diffusion-one-flow/text_encoder.pt"
    text_encoder.load_state_dict(torch.load(parameter_path, map_location='cpu'))
    return text_encoder