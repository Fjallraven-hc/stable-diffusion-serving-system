import sys
import os
from typing import List, Dict

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

class ClipModule(BatchModule):
    def __init__(self, device: str, data_type, parameter_path, tokenizer_config: Dict, text_encoder_config: Dict, **kwargs):
        super().__init__(device=device)
        self.tokenizer_config = tokenizer_config
        self.text_encoder_config = text_encoder_config
        if data_type == "float16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32
        self.parameter_path = parameter_path
        """
        tokenizer_config = {
            "vocab_file": "/path/to/file",
            "merges_file": "/path/to/file",
            "model_max_length": int_value,
        }
        text_encoder_config = {
            "clip_text_config_path": "/path/to/file",
            "parameter_path": "/path/to/file",
            "data_type": "data_type" # [torch.float16, torch.float32]
        }
        """

    def deploy(self, **kwargs):
        from transformers.models.clip.tokenization_clip import CLIPTokenizer
        # init tokenizer, according to config
        self.tokenizer = CLIPTokenizer(**self.tokenizer_config
            #vocab_file=self.tokenizer_config["vocab_file"],
            #merges_file=self.tokenizer_config["merges_file"],
        )
        self.tokenizer.model_max_length=self.tokenizer_config["model_max_length"]

        from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextConfig
        # init text_encoder, according to config
        CLIPTextConfig_path = self.text_encoder_config["clip_text_config_path"]
        self.text_encoder = CLIPTextModel(CLIPTextConfig.from_pretrained(CLIPTextConfig_path))
        
        self.text_encoder.load_state_dict(torch.load(self.parameter_path, map_location="cpu"))
        self.text_encoder = self.text_encoder.to(self.device)
        if self.data_type == torch.float16:
            self.text_encoder = self.text_encoder.half()

        self.deployed = True

    def offload(self, **kwargs):
        # offload model from GPU
        self.text_encoder = self.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        self.deployed = False

    def set_implementation(self, **kwargs):
        # set self.text_encoder data type to fp16, fp32 and etc.
        pass

    def exec_batch(self, batch_request: List[str], **kwargs):
        if not self.deployed:
            raise CustomError("ClipModule is not deployed! Can not exec batch!")
        
        batch_prompt = []
        # form the batch data
        for request in batch_request:
            batch_prompt.append(request["prompt"])

        if type(batch_prompt) != list or type(batch_prompt[0]) != str:
            raise custom_error("ClipModule.exec should input list of str!")
        batch_size = len(batch_prompt)

        text_inputs = self.tokenizer(
            batch_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(batch_prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens: List[str]
        uncond_tokens = [""] * batch_size
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        attention_mask = None
        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * 1, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        for idx in range(len(batch_request)):
            batch_request[idx]["negative_prompt_embeds"] = prompt_embeds[idx:idx+1].cpu()
            batch_request[idx]["prompt_embeds"] = prompt_embeds[batch_size+idx:batch_size+idx+1].cpu()
        return batch_request
