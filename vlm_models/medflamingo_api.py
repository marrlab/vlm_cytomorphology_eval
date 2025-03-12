import requests
from transformers import set_seed

from huggingface_hub import hf_hub_download
import torch
import os
from open_flamingo import create_model_and_transforms
from accelerate import Accelerator
from einops import repeat
from PIL import Image
import sys
from vlm_models.utils import FlamingoProcessor
from vlm_models.utils import clean_generation
import argparse
import pandas as pd
from tqdm import tqdm

set_seed(0)
accelerator = Accelerator()
device = accelerator.device

def load_medflamingo_model(vlm_name='medflamingo'):
    print('Loading model..')

    if vlm_name == 'medflamingo':
        llama_path = "/lustre/groups/labs/marr/qscd01/workspace/furkan.dasdelen/med-flamingo-master/models/llama-7b-hf"

    else:
        raise ValueError(f"Model {vlm_name} not supported")
    
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4
    )
    checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
    print(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    processor = FlamingoProcessor(tokenizer, image_processor)

    # go into eval model and prepare:
    model = accelerator.prepare(model)
    is_main_process = accelerator.is_main_process
    model.eval()

    return model, processor


def medflamingo_api_visual_inquiry(image_path, prompt_text, vlm_name='medflamingo', **kwargs):

    model = kwargs.get('model')
    processor = kwargs.get('processor')
    max_new_tokens = kwargs.get('max_new_tokens', 10000)  # Default to 10000 if not provided

    messages = f"You are a helpful medical assistant. <image>Question: {prompt_text}.\nAnswer:"
    image = Image.open(image_path)
    pixels = processor.preprocess_images([image])
    pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)
    tokenized_data = processor.encode_text(messages)

    generated_text = model.generate(
    vision_x=pixels.to(device),
    lang_x=tokenized_data["input_ids"].to(device),
    attention_mask=tokenized_data["attention_mask"].to(device),
    max_new_tokens=max_new_tokens,
    )
    response = processor.tokenizer.decode(generated_text[0])
    answer = clean_generation(response)

    usage = 0 

    return answer, usage


def medflamingo_api_text_inquiry(prompt_text, vlm_name='medflamingo', **kwargs):
    
    
    model = kwargs.get('model')
    processor = kwargs.get('processor')
    max_new_tokens = kwargs.get('max_new_tokens', 10000)  # Default to 10000 if not provided
    
    messages = prompt_text
    tokenized_data = processor.encode_text(messages)
    generated_text = model.generate(
    lang_x=tokenized_data["input_ids"].to(device),
    attention_mask=tokenized_data["attention_mask"].to(device),
    max_new_tokens=max_new_tokens,
    )
    response = processor.tokenizer.decode(generated_text[0])
    answer = clean_generation(response)

    usage = 0 

    return answer, usage



def medflamingo_multiimage_api_visual_inquiry(image_paths, prompt_texts, vlm_name='medflamingo', **kwargs):

    if len(image_paths) != len(prompt_texts):
        raise ValueError("The number of image paths and prompt texts must be the same.")

    def prepare_messages(prompt_texts, image_paths):

        images = [Image.open(image_path) for image_path in image_paths]
        messages = 'You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question.'       
        for prompt_text, image in zip(prompt_texts, images):
            messages = messages + '<image>' + prompt_text + '\n'        
        return messages, images
    model = kwargs.get('model')
    processor = kwargs.get('processor')
    max_new_tokens = kwargs.get('max_new_tokens', 10000)  # Default to 10000 if not provided

    messages, images = prepare_messages(prompt_texts, image_paths)
    pixels = processor.preprocess_images(images)
    pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)
    tokenized_data = processor.encode_text(messages)

    generated_text = model.generate(
    vision_x=pixels.to(device),
    lang_x=tokenized_data["input_ids"].to(device),
    attention_mask=tokenized_data["attention_mask"].to(device),
    max_new_tokens=max_new_tokens,
    )
    response = processor.tokenizer.decode(generated_text[0])
    answer = clean_generation(response)

    usage = 0 

    return answer, usage