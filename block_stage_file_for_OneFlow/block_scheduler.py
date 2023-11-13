from diffusers.configuration_utils import FrozenDict
from diffusers import EulerAncestralDiscreteScheduler

def SCHEDULER():
    config = [
        ('num_train_timesteps', 1000), ('beta_start', 0.00085), ('beta_end', 0.012), 
        ('beta_schedule', 'scaled_linear'), ('trained_betas', None), 
        ('prediction_type', 'epsilon'), ('skip_prk_steps', True), 
        ('set_alpha_to_one', False), ('steps_offset', 1), 
        ('_class_name', 'PNDMScheduler'), ('_diffusers_version', '0.6.0'), 
        ('clip_sample', False)
    ]
    config ={
        'num_train_timesteps': 1000,
        'beta_start': 0.00085,
        'beta_end': 0.012,
        'beta_schedule': 'scaled_linear',
        'trained_betas': None,
        'prediction_type': 'epsilon',
        'skip_prk_steps': True,
        'set_alpha_to_one': False,
        'steps_offset': 1,
        '_class_name': 'PNDMScheduler',
        '_diffusers_version': '0.6.0',
        'clip_sample': False
    }
    scheduler = EulerAncestralDiscreteScheduler.from_config(config)
    return scheduler
