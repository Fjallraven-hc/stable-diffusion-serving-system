from stable_diffusion_v1_5.stable_diffusion_pipeline import StableDiffusionPipeline
from stable_diffusion_v1_5.stable_diffusion_scheduler import StableDiffusionScheduler
from utils import *
import uuid
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)
#torch.multiprocessing.set_start_method("spawn")

@app.route('/request', methods=['POST'])
def process_request():
    try:
        data = request.json
        print(f"yhc debug:: receive data: {data}")
        data["request_id"] = str(uuid.uuid4())
        request_queue.put(data)

        result_data = None
        while True:
            result_data = request_queue.get()
            if result_data["request_id"] == data["request_id"]:
                result_queue.put(result_data)
            else:
                break

        return jsonify({'result': result_data})
    except KeyError:
        return jsonify({'error': 'Number is missing in the request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    # init pipeline from config
    sd_config_file = "stable_diffusion_v1_5/config.json"
    sd_pipeline = StableDiffusionPipeline(config_path=sd_config_file)

    # init scheduler
    sd_scheduler = StableDiffusionScheduler()
    global request_queue, result_queue
    request_queue = torch.multiprocessing.Queue()
    result_queue = torch.multiprocessing.Queue()
    worker_list, queue_list = sd_scheduler.assign_worker(pipeline=sd_pipeline, head_queue=request_queue, tail_queue=result_queue, test_loop_count=2)
        
    for _worker in worker_list:
        _worker.start()

    app.run(port=5000)

    request = sd_pipeline.task_demo()
    queue_list[0].put(request)
    output = queue_list[-1].get()
    print("output got!")
    print("output:", output)

    for _worker in worker_list:
        _worker.join()
