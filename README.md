# Stable Diffusion Serving System
A low latency & high throughput stable diffusion serving system integrated with most advanced features.
## Features
- SLOs-aware iteration scheduling.
- multi-model/LoRA concurrent serving.
- Co-schedule inference/finetune task.
- Low bit optimization (fp16 for recommend)
- [xFormers](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers), Toolbox to Accelerate Research on transformers, developed by Meta AI.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed), Extreme Speed and Scale for DL Training and Inference, developed by Microsoft Research.
- [OneFlow](https://github.com/Oneflow-Inc/oneflow), a deep learning framework designed to be user-friendly, scalable and efficient.
- Support stable diffusion checkpoints/LoRAs on [civitai](https://civitai.com/).
- Machine Learning Compilation optimization.
## Environment install
For HuggingFace diffusers pipeline, xFormers, DeepSpeed, use `env/base-env.yaml`. Complete DeepSpeed acceleration relies on [CUTLASS](https://github.com/NVIDIA/cutlass) installation.  
For OneFlow, use `env/OneFlow.yaml`, after installation, replace `diffusers.models.unet_2d_condition.forward` function with code in `src/unet_forward_with_different_timesteps.py`.
## Performance
Numbers are collected on Ubuntu 20.04.6 LTS with RTX 4090 24GB, CUDA=11.8.  
All settings use fp16. (Experiment results show that there is no obvious quality loss when using fp16, compared to fp32)  
Inference setting:   
```
{
    "prompt": "an astronaut riding horse on the moon",
    "num_inference_steps": 50,
    "height": 512,
    "width": 512,
    "guidance_scale": 7.5
}
```  
[Note](https://huggingface.co/docs/diffusers/optimization/memory#memory-efficient-attention), If you have PyTorch >= 2.0 installed, you should not expect a speed-up for inference when enabling xformers.
| batch_size | PyTorch=2.1.0+diffusers=0.14.0 | OneFlow=0.9.0 |  xFormers=0.0.22 | DeepSpeed=0.12.2 |
|:----:|:------:|:---:|:---:|:---:|
| 1 | 1.660253 | 0.907718 | 1.837109 | 1.444413 |
| 2 | 2.154117 | 1.481392 | 2.294451 | 2.094967 |
| 4 | 3.949180 | 2.621291 | 4.086211 | 3.907683 |
| 8 | 7.741389 | 5.011853 | 7.610301 | 7.674476 |
