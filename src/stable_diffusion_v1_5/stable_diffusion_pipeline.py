import sys
import os
import json
from typing import Dict, List

script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
sys.path.insert(0, script_dir)

# Add the parent directory to sys.path
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils import *
from .batch_module_list import *

class StableDiffusionPipeline(Pipeline):
    def __init__(self, config_path, **kwargs):
        super().__init__()
        fp = open(config_path, "r")
        config = json.load(fp)

        for module in module_list:
            temp_module = module(**config[module.__name__])
            self.batch_module_list.append(temp_module)
    
    def task_demo(self) -> Dict:
        task = {
                "prompt": "ocean",
                "height": 512,
                "width": 512,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "seed": 0
            }
        return task
    
    def default_deploy(self, **kwargs):
        for module in self.batch_module_list:
            module.deploy()
