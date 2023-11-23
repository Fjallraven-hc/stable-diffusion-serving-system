from diffusers import DiffusionPipeline
import torch
import time

model_path = "/home/yhc/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0"
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
with torch.inference_mode():
    begin = time.perf_counter()
    for prompt in ["a cute cat", "an astronaut riding horse on the moon"]:
        torch.manual_seed(0)
        pipe(prompt).images[0].save(f"{prompt}.jpg")
    print(f"StableDiffusionPipeline latency on batch 1: {(time.perf_counter() - begin) / 2}")


pipe.enable_xformers_memory_efficient_attention()

prompt = "an astronaut riding horse on the moon"
with torch.inference_mode():
    # warmup
    begin = time.perf_counter()
    for prompt in ["a cute cat", "an astronaut riding horse on the moon"]:
        torch.manual_seed(0)
        pipe(prompt)#.images[0].save(f"{prompt}-xFormers.jpg")
    print(f"StableDiffusionPipeline latency on batch 1: {(time.perf_counter() - begin) / 2}")

    for batch_size in [1, 2, 4, 8]:
        temp = time.perf_counter()
        for _ in range(5):
            prompts = [prompt] * batch_size
            torch.manual_seed(0)
            pipe(prompts)
        print(f"yhc debug:: total time used on batch_size: {batch_size} is {(time.perf_counter() - temp)/5}")


# optional: You can disable it via
# pipe.disable_xformers_memory_efficient_attention()