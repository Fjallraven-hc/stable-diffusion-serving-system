import torch
import numpy as np
from typing import Union, List
from .block_unet import UNET, Modified_UNet
from .block_scheduler import SCHEDULER

class unet:
    def __init__(self, device='cpu', dtype=torch.float16):
        self.dtype = dtype
        self.device = device
        self.scheduler = SCHEDULER()
        if dtype == torch.float16:
            self.unet = UNET().to(device).half()
            #self.unet = Modified_UNet().to(device).half()
        else:
            self.unet = UNET().to(device).half()
            #self.unet = Modified_UNet().to(device)
        self.unet.eval()

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
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        sigmas = torch.from_numpy(sigmas).to(device=device)
        timesteps = torch.from_numpy(timesteps).to(device=device)
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
        # 其实就是找出来对应index的sigma，然后scale
        for idx in range(sample.shape[0]):
            sample[idx] /= ((sigma_list[idx] ** 2 + 1) ** 0.5)
            #print(f"yhc debug:: use sigma: {sigma_list[idx]}")
            #print(f"yhc debug:: sample / {((sigma_list[idx]**2 + 1) ** 0.5)}")
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
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # 5. Prepare latent variables
        self.unet_in_channels = 4
        num_channels_latents = self.unet_in_channels
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        if seed is not None:
            torch.manual_seed(seed)
        latents = torch.randn(shape, dtype=dtype, device=device) * self.scheduler.init_noise_sigma
        return latents

    def unet_single_loop(self, prompt_embeds, latents, t, guidance_scale):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        if prompt_embeds.shape[0] != latent_model_input.shape[0]:
            print("Warning! len(prompt_embeds) != len(latents), batch_size is not equal!")
            print(f"yhc debug:: len(prompt_embeds) = {len(prompt_embeds)}")
            print(f"yhc debug:: len(latents) = {len(latents)}")
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents

    def unet_single_loop_different_timesteps(self, prompt_embeds, latents, t, guidance_scale_list, sigma_list, sigma_to_list):
        if t.dim() == 0:
            print("error! t should has the size of len(latents)!")
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.scale_model_input(latent_model_input, torch.cat([sigma_list, sigma_list]))

        if prompt_embeds.shape[0] != latent_model_input.shape[0]:
            print("Warning! len(prompt_embeds) != len(latents), batch_size is not equal!")
            print(f"yhc debug:: len(prompt_embeds) = {len(prompt_embeds)}")
            print(f"yhc debug:: len(latents) = {len(latents)}")
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = []
        for idx in range(len(guidance_scale_list)):
            noise_pred.append(noise_pred_uncond[idx] + guidance_scale_list[idx] * (noise_pred_text[idx] - noise_pred_uncond[idx]))
        noise_pred = torch.stack(noise_pred)
        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler_step(noise_pred, sigma_list, sigma_to_list,latents)
        return latents