import sys
import os
import numpy as np
from typing import List, Dict, Union

# Add the parent directory to sys.path
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
sys.path.insert(0, script_dir)

# Add the grandparent directory to sys.path
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils import *
import torch
torch.set_grad_enabled(False)

class VaeModule(BatchModule):
    def __init__(self, device, data_type, parameter_path, vae_config: Dict, **kwargs):
        super().__init__(device=device)
        if data_type == "float16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32
        self.parameter_path = parameter_path
        self.vae_config = vae_config
        pass

    def deploy(self, **kwargs):
        from diffusers.models.autoencoder_kl import AutoencoderKL
        self.vae = AutoencoderKL(**self.vae_config)
        self.vae.load_state_dict(torch.load(self.parameter_path, map_location='cpu'))
        self.vae = self.vae.to(self.device)
        if self.data_type == torch.float16:
            self.vae.half()
        self.deployed = True
    
    def offload(self, **kwargs):
        # offload model from GPU
        self.vae = self.vae.to("cpu")
        torch.cuda.empty_cache()
        self.deployed = False

    def exec_batch(self, batch_request, **kwargs):
        if not self.deployed:
            raise CustomError("ClipModule is not deployed! Can not exec batch!")
        
        latents = []
        for request in batch_request:
            latents.append(request["latents"])
        latents = torch.cat(latents).to(self.device)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        for idx in range(len(latents)):
            batch_request[idx]["image_numpy_ndarray"] = image[idx:idx+1]
        return batch_request

        