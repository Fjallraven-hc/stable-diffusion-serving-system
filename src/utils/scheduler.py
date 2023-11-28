class Scheduler(object):
    def __init__(self):
        pass
    def set_policy(self, **kwargs):
        # kwargs = keyword arguments
        raise NotImplementedError
    
    def scheduler(self, **kwargs):
        raise NotImplementedError
    
    def generate_batch(self, **kwargs):
        raise NotImplementedError

    def partition_pipeline(self, **kwargs):
        raise NotImplementedError

    def profile_pipeline(self, **kwargs):
        # profile the latency of each module in pipeline
        raise NotImplementedError