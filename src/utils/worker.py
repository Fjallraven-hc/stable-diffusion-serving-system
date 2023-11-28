import os
import torch.multiprocessing as multiprocessing

class Worker(multiprocessing.Process):
    def __init__(self, batch_module_list, input_queue, output_queue, **kwargs):
        super().__init__()
        self.batch_module_list = batch_module_list
        self.input_queue = input_queue
        self.output_queue = output_queue

    def set_device(self, device, **kwargs):
        self.device = device

    def deploy(self, **kwargs):
        # deploy all batch_modules
        for batch_module in self.batch_module_list:
            batch_module.deploy()

    def run(self, **kwargs):
        print("yhc test run.")
        for module in self.batch_module_list:
            module.deploy()
            print(f"pid: {os.getpid()}, serving module: {type(module).__name__}")

        while True:
            if self.input_queue.empty():
                continue
            request = [self.input_queue.get()]
            print(f"pid: {os.getpid()}, serving modules: {self.batch_module_list}, running batch: {request}")
            for module in self.batch_module_list:
                request = module.exec_batch(request)
            #print(request)
            self.output_queue.put(request[0])