tokenizer_config = {
    "vocab_file": "/data/yhc/stable-diffusion-v1-5/tokenizer/vocab.json",
    "merges_file": "/data/yhc/stable-diffusion-v1-5/tokenizer/merges.txt",
    "model_max_length": 77,
}

# clip_text_config_path需要是包含config.json的文件夹路径，而不是config.json的绝对路径
text_encoder_config = {
    "clip_text_config_path": "/data/yhc/stable-diffusion-v1-5/text_encoder",
}

scheduler_config = {
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
unet_config = {
    "sample_size": 64,
    "down_block_types": (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
        ),
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
    "cross_attention_dim": 768,
}
vae_config = {
    "in_channels" : 3,
    "out_channels" : 3,
    "down_block_types": ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
    "up_block_types": ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "act_fn": 'silu',
    "latent_channels": 4,
    "norm_num_groups": 32,
    "scaling_factor": 0.18215
}
feature_extractor_config = {
    "crop_size": 224,
    "do_center_crop": True,
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_resize": True,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "resample": 3,
    "size": 224
}
safety_checker_config = {
    "config_path": "/data/yhc/stable-diffusion-v1-5/safety_checker/config.json",
    "parameter_path": "/data/yhc/stable-diffusion-v1-5/safety_checker/yhc_saved_safety_checker.bin"
}

import json
import torch

config_all = {
    "ClipModule": {
        "device": "cuda",
        "data_type": "float16",
        "tokenizer_config": tokenizer_config,
        "parameter_path": "/data/yhc/stable-diffusion-v1-5/text_encoder/yhc_saved_pytorch_model.bin",
        "text_encoder_config": text_encoder_config,    
    },
    "UNetModule": {
        "device": "cuda",
        "data_type": "float16",
        "scheduler_config": scheduler_config,
        "unet_config": unet_config,
        "parameter_path": "/data/yhc/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin",    
    },
    "VaeModule": {
        "device": "cuda",
        "data_type": "float16",
        "vae_config": vae_config,
        "parameter_path": "/data/yhc/stable-diffusion-v1-5/vae/diffusion_pytorch_model.bin",    
    },
    "SafetyModule": {
        "device": "cuda",
        "data_type": "float16",
        "feature_extractor_config": feature_extractor_config,
        "safety_checker_config": safety_checker_config,
        "parameter_path": "/data/yhc/stable-diffusion-v1-5/safety_checker/yhc_saved_safety_checker.bin"
    }
}
f = open("config.json", "w")
config_load = json.dump(config_all, f, indent=2)

"""for key in config_all.keys():
    print(config_all[key] == config_load[key])
    if not config_all[key] == config_load[key]:
        print(config_all[key])
        print(config_load[key])"""