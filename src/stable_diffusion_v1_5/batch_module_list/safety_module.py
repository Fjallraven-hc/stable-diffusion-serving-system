import sys
import os
from typing import List, Dict

script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
sys.path.insert(0, script_dir)

# Add the parent directory to sys.path
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils import *
import torch

class SafetyModule(BatchModule):
    def __init__(self, device, data_type, parameter_path, feature_extractor_config, safety_checker_config, **kwargs):
        super().__init__(device=device)
        if data_type == "float16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32
        self.parameter_path = parameter_path
        self.feature_extractor_config = feature_extractor_config
        self.safety_checker_config = safety_checker_config

    def deploy(self, **kwargs):
        from transformers.models.clip.image_processing_clip import CLIPImageProcessor
        self.feature_extractor = CLIPImageProcessor(**self.feature_extractor_config)
        self.feature_extractor.feature_extractor_type = "CLIPFeatureExtractor"

        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker, CLIPConfig
        self.safety_checker = StableDiffusionSafetyChecker(CLIPConfig.from_pretrained(self.safety_checker_config["config_path"]))
        self.safety_checker.load_state_dict(torch.load(self.safety_checker_config["parameter_path"], map_location='cpu'))
        self.safety_checker = self.safety_checker.to(self.device)
        if self.data_type == torch.float16:
            self.safety_checker.half()

        self.deployed = True

    def offload(self, **kwargs):
        # offload model from GPU
        self.safety_checker = self.safety_checker.to("cpu")
        torch.cuda.empty_cache()
        self.deployed = False
    
    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        from PIL import Image
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def exec_batch(self, batch_request: List[Dict], **kwargs):
        if not self.deployed:
            raise CustomError("ClipModule is not deployed! Can not exec batch!")
        
        import numpy as np
        image_numpy_ndarray_list = []
        for request in batch_request:
            image_numpy_ndarray_list.append(request["image_numpy_ndarray"])
        images = np.concatenate(image_numpy_ndarray_list, axis=0)

        safety_checker_input = self.feature_extractor(self.numpy_to_pil(images), return_tensors="pt").to(self.device)
        images, has_nsfw_concept = self.safety_checker(
            images=images, clip_input=safety_checker_input.pixel_values.to(self.data_type)
        )
        images = self.numpy_to_pil(images)

        for idx in range(len(batch_request)):
            batch_request[idx]["pillow_image"] = images[idx]
            batch_request[idx]["has_nsfw_concept"] = has_nsfw_concept[idx]
        return batch_request