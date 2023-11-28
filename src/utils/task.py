from typing import Dict

class Task(object):
    def __init__(self, state=None, task_parameter: Dict = None):
        self.state = state
        self.task_parameter = task_parameter
    
    def set_state(self, state):
        self.state = state

    def set_task_parameter(self, task_parameter):
        self.task_parameter = task_parameter
