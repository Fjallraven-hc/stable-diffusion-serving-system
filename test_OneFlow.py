import argparse
from tqdm.auto import tqdm
from onediff.infer_compiler import oneflow_compile
from diffusers import StableDiffusionPipeline
import oneflow as flow
import torch
import time
import numpy as np

from block_stage_file_for_OneFlow import *
#from diffusers.models.unet_2d_condition import UNet2DConditionModel

torch.set_grad_enabled(False)

device = "cuda"

temp = time.perf_counter()
clip_stage = clip(device=device)
print(f"yhc debug: timed used on initialize clip: {time.perf_counter() - temp}")

temp = time.perf_counter()
unet_stage = unet(device=device)
unet_stage.unet = oneflow_compile(unet_stage.unet)
print(f"yhc debug: timed used on initialize unet: {time.perf_counter() - temp}")

temp = time.perf_counter()
vae_and_safety_checker_stage = vae_and_safety_checker(device=device)
print(f"yhc debug: timed used on initialize vae and safety: {time.perf_counter() - temp}")

temp = time.perf_counter()

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

prompt = "an astronaut riding horse on the moon"
height = 512
width = 512
num_inference_steps = 50
guidance_scale = 7.5
with flow.autocast("cuda"):
    for batch_size in [8, 4, 2, 1]:
        prompts = [prompt] * batch_size
        guidance_scale_list = [7.5] * batch_size
        # warmup
        torch.manual_seed(0)
        begin = time.perf_counter()
        prompt_embeds = clip_stage.encode_prompt(prompts)
        latents = unet_stage.prepare_latents(batch_size, 50, 512, 512, device, torch.float16)
        timesteps, sigmas = unet_stage.get_timesteps_and_sigmas(50, device)
        for idx in tqdm(range(50)):
            latents = unet_stage.unet_single_loop_different_timesteps(prompt_embeds, latents, timesteps[idx:idx+1].expand(batch_size), guidance_scale_list, sigmas[idx:idx+1].expand(batch_size), sigmas[idx+1:idx+2].expand(batch_size))
            
        image = vae_and_safety_checker_stage.vae_decode(latents)
        image, safe = vae_and_safety_checker_stage.safety_check(image, device)

        begin = time.perf_counter()
        for _ in range(5):
            torch.manual_seed(0)
            prompt_embeds = clip_stage.encode_prompt(prompts)
            latents = unet_stage.prepare_latents(batch_size, 50, 512, 512, device, torch.float16)
            timesteps, sigmas = unet_stage.get_timesteps_and_sigmas(50, device)
            for idx in tqdm(range(50)):
                latents = unet_stage.unet_single_loop_different_timesteps(prompt_embeds, latents, timesteps[idx:idx+1].expand(batch_size), guidance_scale_list, sigmas[idx:idx+1].expand(batch_size), sigmas[idx+1:idx+2].expand(batch_size))
            image = vae_and_safety_checker_stage.vae_decode(latents)
            image, safe = vae_and_safety_checker_stage.safety_check(image, device)
        print(f"time used on batch {batch_size}: {(time.perf_counter() - begin) / 5}")

"""
with flow.autocast("cuda"):
    torch.manual_seed(0)
    begin = time.perf_counter()
    test_prompt = "this is a test"
    test_latent = unet_stage.prepare_latents(1, 50, 512, 512, device, torch.float16)
    test_timesteps, test_sigmas = unet_stage.get_timesteps_and_sigmas(50, device)
    prompt_embeds = clip_stage.encode_prompt(test_prompt)
    for idx in tqdm(range(50)):
        test_latent = unet_stage.unet_single_loop_different_timesteps(prompt_embeds, test_latent, test_timesteps[idx:idx+1], [7.5], test_sigmas[idx:idx+1], test_sigmas[idx+1:idx+2])
    image = vae_and_safety_checker_stage.vae_decode(test_latent)
    image, safe = vae_and_safety_checker_stage.safety_check(image, device)
    for idx, _image in enumerate(image):
        _image.save(f"{test_prompt}-OneFlow.jpg")
    print(f"time used on batch 1: {time.perf_counter() - begin}")

with flow.autocast("cuda"):
    torch.manual_seed(0)
    begin = time.perf_counter()
    prompt_one = "a cute cat"
    prompt_two = "an astronaut riding horse on the moon"
    prompts = [prompt_one, prompt_two]
    # 这里的num_inference_steps并没有实际意义
    latents = unet_stage.prepare_latents(len(prompts), 50, 512, 512, device, torch.float16)
    timesteps_one, sigmas_one = unet_stage.get_timesteps_and_sigmas(30, device)
    timesteps_two, sigmas_two = unet_stage.get_timesteps_and_sigmas(50, device)
    prompt_embeds = clip_stage.encode_prompt(prompts)
    for idx in tqdm(range(30)):
        timesteps = torch.cat([timesteps_one[idx:idx+1], timesteps_two[idx:idx+1]])
        sigmas = torch.cat([sigmas_one[idx:idx+1], sigmas_two[idx:idx+1]])
        sigmas_to = torch.cat([sigmas_one[idx+1:idx+2], sigmas_two[idx+1:idx+2]])
        latents = unet_stage.unet_single_loop_different_timesteps(prompt_embeds, latents, timesteps, [7.5, 7.5], sigmas, sigmas_to)
    image = vae_and_safety_checker_stage.vae_decode(latents[:1])
    image, safe = vae_and_safety_checker_stage.safety_check(image, device)
    image[0].save(f"{prompts[0]}-OneFlow.jpg")
    prompt_embeds = prompt_embeds[2:]
    latents = latents[1:]
    for idx in tqdm(range(30, 50)):
        timesteps = torch.cat([timesteps_two[idx:idx+1]])
        sigmas = torch.cat([sigmas_two[idx:idx+1]])
        sigmas_to = torch.cat([sigmas_two[idx+1:idx+2]])
        latents = unet_stage.unet_single_loop_different_timesteps(prompt_embeds, latents, timesteps, [7.5], sigmas, sigmas_to)
    image = vae_and_safety_checker_stage.vae_decode(latents)
    image, safe = vae_and_safety_checker_stage.safety_check(image, device)
    image[0].save(f"{prompts[1]}-OneFlow.jpg")
    print(f"time used on batch 1: {time.perf_counter() - begin}")"""