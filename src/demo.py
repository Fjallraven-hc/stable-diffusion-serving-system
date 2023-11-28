from stable_diffusion_v1_5.stable_diffusion_pipeline import StableDiffusionPipeline
from stable_diffusion_v1_5.stable_diffusion_scheduler import StableDiffusionScheduler
from utils import *
import uuid
import torch

if __name__ == "__main__":
    # init pipeline from config
    sd_config_file = "stable_diffusion_v1_5/config.json"
    sd_pipeline = StableDiffusionPipeline(config_path=sd_config_file)

    # init scheduler
    sd_scheduler = StableDiffusionScheduler()
    worker_list, queue_list = sd_scheduler.assign_worker(pipeline=sd_pipeline, test_loop_count=2)
        
    for _worker in worker_list:
        _worker.start()

    request = sd_pipeline.task_demo()
    queue_list[0].put(request)
    output = queue_list[-1].get()
    print("output got!")
    print("output:", output)

    for _worker in worker_list:
        _worker.join()
