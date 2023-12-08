import os
import time
import queue
import torch
import torch.multiprocessing as multiprocessing
import threading

class Worker(multiprocessing.Process):
    def __init__(self, batch_module_list, input_queue, output_queue, **kwargs):
        super().__init__()
        self.batch_module_list = batch_module_list
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_queue = queue.PriorityQueue() # (priority, item). lowest first
        self.current_batch = []
        self.batch_ready = multiprocessing.Semaphore(0)
        self.batch_used = multiprocessing.Semaphore(0)
        self.first_batch = True
        self.loop_module_list = [type(batch_module).__name__ for batch_module in self.batch_module_list if batch_module.loop_module]
        module_name_list = [type(batch_module).__name__ for batch_module in self.batch_module_list]
        self.module_tag = "&".join(module_name_list)
        self.loop_unit = 50 if len(self.loop_module_list) != 0 else 1
        if "device" in kwargs:
            self.device = kwargs["device"]
            for batch_module in self.batch_module_list:
                batch_module.device = self.device
        else:
            self.device = "cuda" # default to cuda:0
        print("-"*10, f"yhc debug:: loop_module_list: {self.loop_module_list}", "-"*10)

    def set_device(self, device, **kwargs):
        self.device = device

    def deploy(self, **kwargs):
        # deploy all batch_modules
        for batch_module in self.batch_module_list:
            batch_module.deploy()

    def schedule_batch(self, **kwargs):
        print(f"pid: [{os.getpid()}], scheduling batch thread working.")
        while True:
            # put request to batch_queue
            while not self.input_queue.empty():
                item = self.input_queue.get()
                item[self.module_tag+"_receive_time"] = time.time()
                # batch_queue item: (priority, request)
                self.batch_queue.put((item["SLO_deadline"] - time.time(), item))

            # empty queue, no need to form batch
            if self.batch_queue.qsize() == 0 and len(self.current_batch) == 0:
                continue

            # avoid concurrent access of self.current_batch
            if self.first_batch:
                self.first_batch = False
            else:
                self.batch_used.acquire()
                print(f"{self.module_tag}, batch_used acquired once.")
            # put finished request to output_queue
            if len(self.current_batch) != 0:
                if len(self.loop_module_list) != 0:
                    for loop_module in self.loop_module_list:
                        for batch in self.current_batch:
                            print("-"*10, f"loop_index: {batch[1]['loop_index'][loop_module]}", "-"*10)
                            if batch[1]["loop_index"][loop_module] >= batch[1]["loop_num"][loop_module]:
                                batch[1][self.module_tag+"_send_time"] = time.time()
                                self.output_queue.put(batch[1])
                            else:
                                self.batch_queue.put(batch)
                # no loop module
                else:
                    for batch in self.current_batch:
                        batch[1][self.module_tag+"_send_time"] = time.time()
                        self.output_queue.put(batch[1])
                    
            # form new batch
            # test policy, put the head one
            if self.batch_queue.qsize() > 0:
                self.current_batch = [self.batch_queue.get()]
            else:
                self.current_batch = []
            self.batch_ready.release()

    def run(self, **kwargs):
        torch.set_grad_enabled(False)
        with torch.inference_mode():
            try:
                #print("yhc test run.")
                print(f"pid: [{os.getpid()}], module list: {self.batch_module_list}")
                for module in self.batch_module_list:
                    module.device = self.device
                    module.deploy()
                    print(f"pid: [{os.getpid()}], serving module: {type(module).__name__}")

                schedule_batch_thread = threading.Thread(target=self.schedule_batch)
                schedule_batch_thread.start()
                
                # sequentially run the batch module
                while True:
                    self.batch_ready.acquire()
                    print(f"{self.module_tag}, batch_ready acquired once.")
                    if len(self.current_batch) == 0:
                        self.batch_used.release()
                        continue
                    # wipe off the priority
                    batch_request = [batch[1] for batch in self.current_batch]
                    # execute through the pipeline
                    for _ in range(self.loop_unit):
                        for module in self.batch_module_list:
                            batch_request = module.exec_batch(batch_request)
                        #print(batch_request)
                    # update the metadata
                    for idx in range(len(batch_request)):
                        for loop_module in self.loop_module_list:
                            batch_request[idx]["loop_index"][loop_module] += self.loop_unit
                        self.current_batch[idx] = (self.current_batch[idx][0], batch_request[idx])
                    self.batch_used.release()
                    """if self.input_queue.empty():
                        continue
                    request = [self.input_queue.get()]
                    print(f"pid: [{os.getpid()}], serving modules: {self.batch_module_list}, running batch length: {len(request)}")
                    for module in self.batch_module_list:
                        request = module.exec_batch(request)
                    #print(request)
                    self.output_queue.put(request[0])"""
            # Worker process receive interrupt
            except KeyboardInterrupt:
                print(f"Child process:[{os.getpid()}] received KeyboardInterrupt.")