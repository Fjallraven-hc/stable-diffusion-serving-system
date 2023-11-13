from tqdm.auto import tqdm
import torch
import time
import numpy as np

from block_stage_file import *
#from diffusers.models.unet_2d_condition import UNet2DConditionModel

torch.set_grad_enabled(False)

device = "cuda"

temp = time.perf_counter()
clip_stage = clip(device=device)
print(f"yhc debug: timed used on initialize clip: {time.perf_counter() - temp}")

temp = time.perf_counter()
unet_stage = unet(device=device)
print(f"yhc debug: timed used on initialize unet: {time.perf_counter() - temp}")

temp = time.perf_counter()
vae_and_safety_checker_stage = vae_and_safety_checker(device=device)
print(f"yhc debug: timed used on initialize vae and safety: {time.perf_counter() - temp}")

with torch.inference_mode():
    begin = time.perf_counter()
    prompt = "a cute cat"
    prompt_embeds = clip_stage.encode_prompt(prompt)
    latents = unet_stage.prepare_latents(1, 50, 512, 512, device, torch.float16, seed=0)
    for t in unet_stage.scheduler.timesteps:
        latents = unet_stage.unet_single_loop(prompt_embeds, latents, t, guidance_scale=7.5)
    image = vae_and_safety_checker_stage.vae_decode(latents)
    images, safe = vae_and_safety_checker_stage.safety_check(image, device)
    images[0].save(f"{prompt}.jpg")
    print(f"pipeline latency on batch 1: {time.perf_counter() - begin}")

import deepspeed
deepspeed.init_inference(
    model=clip_stage.text_encoder,      # Transformers models
    mp_size=1,        # Number of GPU
    dtype=torch.float16, # dtype of the weights (fp16)
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=False, # replace the model with the kernel injector
)
deepspeed.init_inference(
    model=unet_stage.unet,      # Transformers models
    mp_size=1,        # Number of GPU
    dtype=torch.float16, # dtype of the weights (fp16)
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=False, # replace the model with the kernel injector
)

prompt = "an astronaut riding horse on the moon"
with torch.inference_mode():
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


def measure_latency(pipe, prompt):
    latencies = []
    # warm up
    pipe.set_progress_bar_config(disable=True)
    for _ in range(2):
        _ =  pipe(prompt)
    # Timed run
    for _ in range(10):
        start_time = time.perf_counter()
        _ = pipe(prompt)
        latency = time.perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_s = np.mean(latencies)
    time_std_s = np.std(latencies)
    time_p95_s = np.percentile(latencies,95)
    return f"P95 latency (seconds) - {time_p95_s:.2f}; Average latency (seconds) - {time_avg_s:.2f} +\- {time_std_s:.2f};", time_p95_s
"""
from diffusers import StableDiffusionPipeline
model_path = "/home/yhc/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0"
torch.manual_seed(0)
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")
for prompt in ["a cute cat", "an astronaut riding horse on the moon"]:
    result = pipe(prompt)
    result

with torch.inference_mode():
    prompt = "a photo of an astronaut riding a horse on mars"
    ds_results = measure_latency(pipe, prompt)
    print(f"StableDiffusionPipeline model: {ds_results[0]}") 

prompt = "a photo of an astronaut riding a horse on mars"
with torch.inference_mode():
    torch.manual_seed(0)
    deepspeed.init_inference(
        model=getattr(pipe,"model", pipe),      # Transformers models
        mp_size=1,        # Number of GPU
        dtype=torch.float16, # dtype of the weights (fp16)
        replace_method="auto", # Lets DS autmatically identify the layer to replace
        replace_with_kernel_inject=False, # replace the model with the kernel injector
    )
    for prompt in ["a cute cat", "an astronaut riding horse on the moon"]:
        pipe(prompt).images[0].save(f"{prompt}-DeepSpeed.jpg")
    ds_results = measure_latency(pipe, prompt)
    print(f"DeepSpeed model: {ds_results[0]}")

#print("DeepSpeed Inference Engine initialized")"""