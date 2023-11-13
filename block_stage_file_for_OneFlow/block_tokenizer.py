import torch
from transformers.models.clip.tokenization_clip import CLIPTokenizer

def TOKENIZER():
    tokenizer = CLIPTokenizer(
        vocab_file="/data/yhc/stable-diffusion-one-flow/vocab.json",
        merges_file="/data/yhc/stable-diffusion-one-flow/merges.txt",
    )
    tokenizer.model_max_length=77
    return tokenizer
