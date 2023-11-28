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

import time
import torch
from utils import *
from .batch_module_list import *
from .stable_diffusion_pipeline import StableDiffusionPipeline

class StableDiffusionScheduler(Scheduler):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def profile_pipeline(self, pipeline: StableDiffusionPipeline, test_loop_count = 10, **kwargs) -> Dict:
        profile_latency = []
        task_demo = [pipeline.task_demo()]
        for module in pipeline.batch_module_list:
            module.deploy()
            print(f"profile {type(module).__name__} is ongoing.")

            if not module.loop_module:
                begin = time.perf_counter()
                for _ in range(test_loop_count):
                    task_demo = module.exec_batch(task_demo)
                profile_latency.append((time.perf_counter() - begin) / test_loop_count)
            else:
                begin = time.perf_counter()
                for _ in range(test_loop_count):
                    for __ in range(module.avg_loop_count):
                        task_demo = module.exec_batch(task_demo)
                        # necessary to set loop_index
                        task_demo[0]["loop_index"] = 0
                profile_latency.append((time.perf_counter() - begin) / test_loop_count)
            # offload module
            module.offload()
            torch.cuda.empty_cache()
        
        return profile_latency
    
    def partition_pipeline(self, profile_data: List[float], **kwargs):
        pipeline_length = len(profile_data)
        # 使用dp计算最佳切分
        # 计算前缀和
        prefix_sum = [0]
        for num in profile_data:
            prefix_sum.append(prefix_sum[-1] + num)

        # 初始化动态规划数组，dp[i][j] 表示将前 i 个元素划分为 j 个子列表时的最小元素和
        dp = [[float('inf')] * (pipeline_length + 1) for _ in range(pipeline_length + 1)]
        dp[0][0] = 0  # 0 个子列表时，元素和为 0

        # record数组
        record = [[-1] * (pipeline_length + 1) for _ in range(pipeline_length + 1)]

        # 递推计算动态规划数组
        for i in range(1, pipeline_length + 1):
            for j in range(1, i + 1):
                for k in range(i):
                    # 从前 k 个元素到第 i 个元素构成一个子列表
                    if dp[i][j] > max(dp[k][j - 1], prefix_sum[i] - prefix_sum[k]):
                        record[i][j] = k
                    dp[i][j] = min(dp[i][j], max(dp[k][j - 1], prefix_sum[i] - prefix_sum[k]))
        
        # 找到最小元素和对应的子列表划分
        min_element_sum = min(dp[pipeline_length])
        j = dp[pipeline_length].index(min_element_sum)

        i = pipeline_length
        partitions = []
        while j > 0:
            k = record[i][j]
            sublist = [e for e in range(k+1, i+1)]
            partitions.append(sublist)
            i -= len(sublist)
            j -= 1
        partitions.reverse()
        return min_element_sum, partitions
    
    def assign_worker(self, pipeline, head_queue=None, tail_queue=None, test_loop_count=5, **kwargs):
        profile_data = self.profile_pipeline(pipeline=pipeline, test_loop_count=test_loop_count)
        min_max_stage_latency, partitions = self.partition_pipeline(profile_data)

        worker_nums = len(partitions)
        queue_list = [torch.multiprocessing.Queue() for _ in range(worker_nums + 1)]
        #queue_list.insert(0, head_queue)
        #queue_list.append(tail_queue)

        worker_list = []
        for idx in range(len(partitions)):
            #  each partition correspond to a worker
            batch_module_list = []
            for module_id in partitions[idx]:
                batch_module_list.append(pipeline.batch_module_list[module_id-1])
            worker_list.append(Worker(batch_module_list=batch_module_list,
                                    input_queue=queue_list[idx],
                                    output_queue=queue_list[idx+1]))
        return worker_list, queue_list
        
    def generate_batch(self, **kwargs):
        return super().generate_batch(**kwargs)
        
