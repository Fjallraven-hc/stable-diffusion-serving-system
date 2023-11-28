"""
This script shows a naive way, may be not so elegant, to load Lora (safetensors) weights in to diffusers model

For the mechanism of Lora, please refer to https://github.com/cloneofsimo/lora

Copyright 2023: Haofan Wang, Qixun Wang
"""
import time
import torch
from yhc_sd_pipeline.block_stage_file import *
from safetensors.torch import load_file

from lora_diffusion import tune_lora_scale, patch_pipe
from lora_diffusion.lora import LoraInjectedLinear, LoraInjectedConv2d

torch.set_grad_enabled(False)

if __name__ == "__main__":
    # load diffusers model
    device = "cuda:1"
    clip_stage = clip(device)
    unet_stage = unet(device)
    vae_and_safety_stage = vae_and_safety_checker(device)

    # load lora weight
    lora_path = "/home/yhc/lora/civitai_loras/wanostyle_2_offset.safetensors"
    state_dict = load_file(lora_path, device=device)

    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'

    alpha = 0.75

    visited = []

    layer_type = {}
    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue
        if 'text' in key:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
            curr_layer = clip_stage.text_encoder
        else:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
            curr_layer = unet_stage.unet
        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            #print(temp_name)
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                print(f"name: {temp_name} succeed!")
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
                #print(f"name: {temp_name} succeed!")
            except Exception:
                #print(f"name: {temp_name} failed!")
                if len(temp_name) > 0:
                    temp_name += '_'+layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # 已经定位到了注入LoRA的layer
        if curr_layer.__class__.__name__ == "Linear":
            replace_layer = LoraInjectedLinear(curr_layer.in_features, curr_layer.out_features)
            replace_layer.linear.weight = curr_layer.weight
            curr_layer = replace_layer
        elif curr_layer.__class__.__name__ == "Conv2d":
            replace_layer = LoraInjectedConv2d(
                curr_layer.in_channels,
                curr_layer.out_channels,
                curr_layer.kernel_size,
                curr_layer.padding,
                curr_layer.dilation,
                curr_layer.groups,
                curr_layer.bias
            )
            replace_layer.conv.weight = curr_layer.weight
            curr_layer = replace_layer
            pass
        
        # print(type(curr_layer))
        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))
        # pair_keys = ["lora_up", "lora_down"]
        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            # Conv2d case
            weight_up = state_dict[pair_keys[0]]#.squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]]#.squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.lora_up_list.append(weight_up)
            curr_layer.lora_down_list.append(weight_down)
            #curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            # Linear case
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.lora_up_list.append(weight_up)
            curr_layer.lora_down_list.append(weight_down)
            #curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
        # update visited list
        for item in pair_keys:
            visited.append(item)

def patch_lora_to_stable_diffusion(lora_path, text_encoder, unet):
    lora_path = "/home/yhc/lora/civitai_loras/wanostyle_2_offset.safetensors"
    state_dict = load_file(lora_path, device=text_encoder.device)

    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'

    alpha = 0.75
    visited = []

    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue
        if 'text' in key:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
            curr_layer = text_encoder
        else:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
            curr_layer = unet
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                print(f"name: {temp_name} succeed!")
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += '_'+layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # 已经定位到了注入LoRA的layer
        if curr_layer.__class__.__name__ == "Linear":
            replace_layer = LoraInjectedLinear(curr_layer.in_features, curr_layer.out_features)
            replace_layer.linear.weight = curr_layer.weight
            curr_layer = replace_layer
        elif curr_layer.__class__.__name__ == "Conv2d":
            replace_layer = LoraInjectedConv2d(
                curr_layer.in_channels,
                curr_layer.out_channels,
                curr_layer.kernel_size,
                curr_layer.padding,
                curr_layer.dilation,
                curr_layer.groups,
                curr_layer.bias
            )
            replace_layer.conv.weight = curr_layer.weight
            curr_layer = replace_layer
            pass
        
        # print(type(curr_layer))
        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))
        # pair_keys = ["lora_up", "lora_down"]
        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            # Conv2d case
            weight_up = state_dict[pair_keys[0]]#.squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]]#.squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.lora_up_list.append(weight_up)
            curr_layer.lora_down_list.append(weight_down)
            #curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            # Linear case
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.lora_up_list.append(weight_up)
            curr_layer.lora_down_list.append(weight_down)
            #curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
        # update visited list
        for item in pair_keys:
            visited.append(item)
###
"""torch.manual_seed(637315696)

# 2. Forward embeddings and negative embeddings through text encoder
prompt = '1boy, wanostyle, monkey d luffy, smiling, straw hat, looking at viewer, solo, upper body, ((masterpiece)), (best quality), (extremely detailed), depth of field, sketch, dark intense shadows, sharp focus, soft lighting, hdr, colorful, good composition, fire all around, spectacular, closed shirt, anime screencap, scar under eye, ready to fight, black eyes'
negative_prompt = '(painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username artist name, (worst quality, low quality:1.4), bad anatomy, watermark, signature, text, logo'
max_length = clip_stage.tokenizer.model_max_length

input_ids = clip_stage.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids
input_ids = input_ids.to(device)

negative_ids = clip_stage.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
negative_ids = negative_ids.to(device)

concat_embeds = []
neg_embeds = []
for i in range(0, input_ids.shape[-1], max_length):
    concat_embeds.append(clip_stage.text_encoder(input_ids[:, i: i + max_length])[0])
    neg_embeds.append(clip_stage.text_encoder(negative_ids[:, i: i + max_length])[0])

prompt_embeds = torch.cat(concat_embeds, dim=1)
negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

embeds = torch.cat([prompt_embeds, negative_prompt_embeds])

latents = unet_stage.prepare_latents(1, 30, 512, 512, device, torch.float16)
for t in unet_stage.scheduler.timesteps:
    latents = unet_stage.unet_single_loop(embeds, latents, t, guidance_scale=9)

image = vae_and_safety_stage.vae_decode(latents)
image, safe = vae_and_safety_stage.safety_check(image, device)

"""