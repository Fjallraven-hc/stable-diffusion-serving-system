class BatchModule(object):
    def __init__(self, device):
        self.device = device
        self.deployed = False
        self.loop_module = False
        self.avg_loop_count = -1

    def set_device(self, device):
        self.device = device

    def set_avg_loop_count(self, avg_loop_count: int):
        self.avg_loop_count = avg_loop_count

    def deploy(self, **kwargs):
        raise NotImplementedError

    def set_implementation(self, **kwargs):
        raise NotImplementedError
    
    def exec_batch(self, batch_request, **kwargs):
        raise NotImplementedError