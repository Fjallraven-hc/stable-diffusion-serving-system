import uuid
import io
import time
import torch
import numpy
import PIL
import asyncio
from aiohttp import web
import json
from stable_diffusion_v1_5.stable_diffusion_pipeline import StableDiffusionPipeline
from stable_diffusion_v1_5.stable_diffusion_scheduler import StableDiffusionScheduler
from utils import *


def worker(input_queue, output_queue):
    while True:
        # receive task
        data = input_queue.get()
        if data is None:
            break  # 结束信号

        # 模拟数据处理
        result = {"status": "processed", "original": data}

        # 发送处理结果
        output_queue.put(result)

async def handle_request(request):
    data = await request.json()
    data["receive_time"] = time.time()
    data["SLO_deadline"] = data["receive_time"] + data["SLO"]

    # 发送数据到工作队列
    input_queue.put(data)

    # 从结果队列获取响应
    result = await asyncio.get_running_loop().run_in_executor(None, output_queue.get)
    result["send_time"] = time.time()
    for key in result.keys():
        #print(f"key: {key}, value type: {type(result[key])}")
        if type(result[key]) == torch.Tensor:
            result[key] = result[key].tolist()
        if type(result[key]) == numpy.ndarray:
            result[key] = result[key].tolist()
        if type(result[key]) == PIL.Image.Image:
            numpy_array = numpy.array(result[key])
            result[key] = numpy_array.tolist()
            
    # 返回HTTP响应
    return web.Response(text=json.dumps(result))

if __name__ == "__main__":
    # no need to pass gradient
    torch.set_grad_enabled(False)
    try:
        # init pipeline from config
        sd_config_file = "stable_diffusion_v1_5/config.json"
        sd_pipeline = StableDiffusionPipeline(config_path=sd_config_file)

        # init scheduler
        sd_scheduler = StableDiffusionScheduler()

        # init queue
        worker_nums = 3
        # one interesting observation
        # multiprocessing.Manager().Queue() is much more efficient than multiprocessing.Queue()
        input_queue = torch.multiprocessing.Manager().Queue()
        output_queue = torch.multiprocessing.Manager().Queue()
        queue_list = [torch.multiprocessing.Manager().Queue() for _ in range(worker_nums - 1)]
        queue_list.insert(0, input_queue)
        queue_list.append(output_queue)        

        # 创建工作进程
        worker_list = []
        worker_list.append(Worker(batch_module_list=sd_pipeline.batch_module_list[0:1], input_queue=queue_list[0], output_queue=queue_list[1], device="cuda:0"))
        worker_list.append(Worker(batch_module_list=sd_pipeline.batch_module_list[1:2], input_queue=queue_list[1], output_queue=queue_list[2], device="cuda:1"))
        worker_list.append(Worker(batch_module_list=sd_pipeline.batch_module_list[2:], input_queue=queue_list[2], output_queue=queue_list[3], device="cuda:2"))

        for _worker in worker_list:
            _worker.start()

        # 设置异步HTTP服务器
        app = web.Application()
        app.router.add_post('/inference', handle_request)
        loop = asyncio.get_event_loop()

        # 运行HTTP服务器
        web.run_app(app, port=8080)
        
        for _worker in worker_list:
            _worker.join()
    except KeyboardInterrupt:
        print("-"*10,"Main process received KeyboardInterrupt","-"*10)