import torch
from transformers.models.clip.tokenization_clip import CLIPTokenizer

def TOKENIZER():
    tokenizer = CLIPTokenizer(
        vocab_file="/data/yhc/stable-diffusion-v1-5/tokenizer/vocab.json",
        merges_file="/data/yhc/stable-diffusion-v1-5/tokenizer/merges.txt",
    )
    tokenizer.model_max_length=77
    return tokenizer
