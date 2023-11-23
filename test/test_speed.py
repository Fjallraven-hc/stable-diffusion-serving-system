from tqdm.auto import tqdm
import time
import torch
from PIL import Image
import sys
import os

# Get the absolute path to the current script
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
# Add the parent directory to sys.path
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from block_stage_file import *

torch.set_grad_enabled(False)

device = "cuda"
prompt = "a cute cat"
prompt = "an astronaut riding horse on the moon"
prompts = [prompt] * 1
height = 512
width = 512
num_inference_steps = 50
guidance_scale = [7.5, 6.5]
guidance_scale = 7.5

temp = time.perf_counter()
clip_stage = clip(device=device)
print(f"yhc debug: timed used on initialize clip: {time.perf_counter() - temp}")
temp = time.perf_counter()
unet_stage = unet(device=device)
print(f"yhc debug: timed used on initialize unet: {time.perf_counter() - temp}")
temp = time.perf_counter()
vae_and_safety_checker_stage = vae_and_safety_checker(device=device)
print(f"yhc debug: timed used on initialize vae and safety: {time.perf_counter() - temp}")
temp = time.perf_counter()

for batch_size in [1, 2, 4, 8]:
    temp = time.perf_counter()
    for _ in range(5):
        prompts = [prompt] * batch_size
        prompt_embeds = clip_stage.encode_prompt(prompts)
        torch.manual_seed(0)
        latents = unet_stage.prepare_latents(batch_size, num_inference_steps, height, width, "cuda:0", torch.float16)
        for t in tqdm(unet_stage.scheduler.timesteps):
            latents = unet_stage.unet_single_loop(prompt_embeds, latents, t, guidance_scale)
        image = vae_and_safety_checker_stage.vae_decode(latents)
        image, safe = vae_and_safety_checker_stage.safety_check(image, "cuda:0")
    print(f"yhc debug:: total time used on batch_size: {batch_size} is {(time.perf_counter() - temp)/5}")

    """for idx, _image in enumerate(image):
        _image.save(f"yhc_gaga_{idx}.jpg")"""