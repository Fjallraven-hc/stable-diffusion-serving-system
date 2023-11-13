import torch
from PIL import Image
from .block_vae import VAE
from .block_feature_extractor import FEATURE_EXTRACTOR
from .block_safety_checker import SAFETY_CHECKER

class vae_and_safety_checker:
    def __init__(self, device, dtype=torch.float16):
        self.feature_extractor = FEATURE_EXTRACTOR()
        self.device = device
        self.dtype = dtype
        if dtype == torch.float16:
            self.vae = VAE().to(device).half()
            self.safety_checker = SAFETY_CHECKER().to(device).half()
        else:
            self.vae = VAE().to(device)
            self.safety_checker = SAFETY_CHECKER().to(device)
        self.vae.eval()
        self.safety_checker.eval()

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def vae_decode(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    def safety_check(self, image, device):
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
        image, has_nsfw_concept = self.safety_checker(
            images=image, clip_input=safety_checker_input.pixel_values.to(self.dtype)
        )
        return self.numpy_to_pil(image), has_nsfw_concept

