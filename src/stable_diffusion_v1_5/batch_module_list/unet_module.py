import sys
import os
import numpy as np
from typing import List, Dict, Union

# Add the parent directory to sys.path
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
sys.path.insert(0, script_dir)

# Add the grandparent directory to sys.path
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from utils import *
import torch
torch.set_grad_enabled(False)

class UNetModule(BatchModule):
    def __init__(self, device, data_type, parameter_path, scheduler_config: Dict, unet_config: Dict, **kwargs):
        super().__init__(device=device)
        self.scheduler_config = scheduler_config
        self.loop_module = True
        self.avg_loop_count = kwargs["avg_loop_count"]
        self.unet_config = unet_config
        if data_type == "float16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32
        self.parameter_path =  parameter_path
        """
        config demo:
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
            "parameter_path": "/path/to/parameter"
        }
        """

    def deploy(self, **kwargs):
        from diffusers import EulerAncestralDiscreteScheduler
        # init scheduler
        self.scheduler = EulerAncestralDiscreteScheduler.from_config(self.scheduler_config)

        from diffusers.models.unet_2d_condition import UNet2DConditionModel
        from .modified_unet_2d_condition import Modified_UNet2DConditionModel
        # init unet
        self.unet = Modified_UNet2DConditionModel(**self.unet_config)
        self.unet.load_state_dict(torch.load(self.parameter_path, map_location='cpu'))
        self.unet = self.unet.to(self.device)
        if self.data_type == torch.float16:
            self.unet.half()
        self.deployed =  True

    def offload(self, **kwargs):
        # offload model from GPU
        self.unet = self.unet.to("cpu")
        torch.cuda.empty_cache()
        self.deployed = False

    def get_timesteps_and_sigmas(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        # added by yhc, to give a request with its relevant sigmas and timesteps
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        timesteps = np.linspace(0, self.scheduler.config.num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
        sigmas = np.array(((1 - self.scheduler.alphas_cumprod) / self.scheduler.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        if self.data_type == torch.float16:
            sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float16)
            timesteps = torch.from_numpy(timesteps).to(device=self.device).to(torch.float16)
        else:
            sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
            timesteps = torch.from_numpy(timesteps).to(device=self.device).to(torch.float32)
        sigmas = torch.from_numpy(sigmas).to(device=self.device)
        return timesteps, sigmas

    def scale_model_input(
        self, sample: torch.FloatTensor, sigma_list: Union[List[float], torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`float` or `torch.FloatTensor`): the current timestep in the diffusion chain

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        # sample.shape[0] == timestep.shape[0]
        # 找出来对应index的sigma，然后scale
        for idx in range(sample.shape[0]):
            sample[idx] /= ((sigma_list[idx] ** 2 + 1) ** 0.5)
        return sample

    def scheduler_step(
        self,
        model_output: torch.FloatTensor,
        sigma_list: Union[List[float], torch.FloatTensor],
        sigma_to_list: Union[List[float], torch.FloatTensor],
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        output_list = []
        for idx in range(model_output.shape[0]):
            if self.scheduler.config.prediction_type == "epsilon":
                pred_original_sample = sample[idx] - sigma_list[idx] * model_output[idx]
            elif self.scheduler.config.prediction_type == "v_prediction":
                # * c_out + input * c_skip
                pred_original_sample = model_output[idx] * (-sigma_list[idx] / (sigma_list[idx]**2 + 1) ** 0.5) + (sample[idx] / (sigma_list[idx]**2 + 1))
            
            sigma_from = sigma_list[idx]
            sigma_to = sigma_to_list[idx]
            sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
            sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

            # 2. Convert to an ODE derivative
            derivative = (sample[idx] - pred_original_sample) / sigma_list[idx]

            dt = sigma_down - sigma_list[idx]

            prev_sample = sample[idx] + derivative * dt

            device = model_output.device

            noise = torch.randn(model_output[idx].shape, dtype=model_output.dtype, device=device)

            prev_sample = prev_sample + noise * sigma_up

            output_list.append(prev_sample)
        return torch.stack(output_list)

    def prepare_latents(self, batch_size, num_inference_steps, height, width, device, dtype, seed=None):
        #self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.unet_in_channels = 4
        num_channels_latents = self.unet_in_channels
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        if seed is not None:
            torch.manual_seed(seed)
        latents = torch.randn(shape, dtype=dtype, device=device) * self.scheduler.init_noise_sigma
        return latents

    def exec_batch(self, batch_request: List[Dict], **kwargs):
        if not self.deployed:
            raise CustomError("ClipModule is not deployed! Can not exec batch!")
        
        for request in batch_request:
            if "loop_index" not in request:
                request["loop_index"] = 0
                request["latents"] = self.prepare_latents(
                    batch_size=1,
                    num_inference_steps=request["num_inference_steps"],
                    height=request["height"],
                    width=request["width"],
                    device=self.device,
                    dtype=self.data_type,
                    seed=request["seed"]
                )
                timestamps, sigma_list = self.get_timesteps_and_sigmas(
                    num_inference_steps=request["num_inference_steps"], 
                    device=self.device
                    )
                request["timestamps"] = timestamps
                request["sigma_list"] = sigma_list
        
        negative_prompt_embeds_list = []
        prompt_embeds_list = []
        latents_list = []
        timestamps_list= []
        guidance_scale_list = []
        sigma_list = []
        sigma_to_list = []

        for request in batch_request:
            negative_prompt_embeds_list.append(request["negative_prompt_embeds"])
            prompt_embeds_list.append(request["prompt_embeds"])
            latents_list.append(request["latents"])
            timestamps_list.append(request["timestamps"][request["loop_index"]:request["loop_index"] + 1])
            guidance_scale_list.append(request["guidance_scale"])
            sigma_list.append(request["sigma_list"][request["loop_index"]:request["loop_index"] + 1])
            sigma_to_list.append(request["sigma_list"][request["loop_index"] + 1:request["loop_index"] + 2])

        # 规定multi-dim的tensor，用torch.cat聚合
        prompt_embeds = torch.cat(
            [torch.cat(negative_prompt_embeds_list),
             torch.cat(prompt_embeds_list)]
            ).to(self.device)
        latents = torch.cat(latents_list).to(self.device)
        timestamps = torch.cat(timestamps_list).to(self.device)
        sigma_list = torch.cat(sigma_list).to(self.device)
        sigma_to_list = torch.cat(sigma_to_list).to(self.device)

        if timestamps.dim() == 0:
            print("error! t should has the size of len(latents)!")
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.scale_model_input(latent_model_input, torch.cat([sigma_list, sigma_list]))

        if prompt_embeds.shape[0] != latent_model_input.shape[0]:
            print("Warning! len(prompt_embeds) != len(latents), batch_size is not equal!")
            print(f"yhc debug:: len(prompt_embeds) = {len(prompt_embeds)}")
            print(f"yhc debug:: len(latents) = {len(latent_model_input)}")
        noise_pred = self.unet(
            latent_model_input,
            timestamps,
            encoder_hidden_states=prompt_embeds,
        ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = []
        for idx in range(len(guidance_scale_list)):
            noise_pred.append(noise_pred_uncond[idx] + guidance_scale_list[idx] * (noise_pred_text[idx] - noise_pred_uncond[idx]))
        noise_pred = torch.stack(noise_pred)
        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler_step(noise_pred, sigma_list, sigma_to_list,latents)

        for idx in range(len(batch_request)):
            batch_request[idx]["latents"] = latents[idx:idx+1].cpu()
            batch_request[idx]["loop_index"] += 1
            batch_request[idx]["timestamps"] = batch_request[idx]["timestamps"].cpu()
            batch_request[idx]["sigma_list"] = batch_request[idx]["sigma_list"].cpu()
        return batch_request
