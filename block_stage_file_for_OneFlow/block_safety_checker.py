import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker, CLIPConfig

def SAFETY_CHECKER():
    config_path = "/data/yhc/stable-diffusion-v1-5/safety_checker/config.json"
    parameter_path = "/data/yhc/stable-diffusion-one-flow/safety_checker.pt"
    safety_checker = StableDiffusionSafetyChecker(CLIPConfig.from_pretrained(config_path))
    safety_checker.load_state_dict(torch.load(parameter_path, map_location='cpu'))
    return safety_checker