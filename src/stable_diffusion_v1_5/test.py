from stable_diffusion_pipeline import StableDiffusionPipeline

config_path = "config.json"
p = StableDiffusionPipeline(config_path=config_path)
for module in p.batch_module_list:
    module.deploy()
request = [{
    "prompt": "ocean",
    "height": 512,
    "width": 512,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 0
    }]
p.batch_module_list[0].exec(request)
for _ in range(50):
    p.batch_module_list[1].exec(request)
p.batch_module_list[2].exec(request)
p.batch_module_list[3].exec(request)
