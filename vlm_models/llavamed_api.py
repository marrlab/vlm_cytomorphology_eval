import argparse
import torch
import os
import json
import pandas as pd

import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging
from tqdm import tqdm


def load_llavamed_model(vlm_name='llavamed'):

    device = "cuda"
    if vlm_name == 'llavamed':
        model_path = "/lustre/groups/labs/marr/qscd01/workspace/furkan.dasdelen/vlm_cytomorphology_eval/LLaVA-Med-main/llava-med-v1.5-mistral-7b"
    else:
        raise ValueError(f"Model {vlm_name} not supported")
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device=device)

   
    return model, tokenizer, image_processor



def llavamed_api_visual_inquiry(image_path, prompt_text, vlm_name='llavamed', **kwargs): 
    model = kwargs.get('model')
    image_processor = kwargs.get('image_processor')
    tokenizer = kwargs.get('tokenizer')
    max_new_tokens = kwargs.get('max_new_tokens', 10000) 
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model.config.mm_use_im_start_end:
        prompt_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt_text
    else:
        prompt_text = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text

    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)[0]

    temperature = 0.2
    top_p = None
    num_beams = 1
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_new_tokens,
            use_cache=True)

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    usage = 0 

    return answer, usage

def llavamed_api_text_inquiry(prompt_text, vlm_name='llavamed', **kwargs):
    model = kwargs.get('model')
    image_processor = kwargs.get('image_processor')
    tokenizer = kwargs.get('tokenizer')
    max_new_tokens = kwargs.get('max_new_tokens', 10000) 
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    temperature = 0.2
    top_p = None
    num_beams = 1
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_new_tokens,
            use_cache=True)

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    usage = 0

    return answer, usage


def llavamed_multiimage_api_visual_inquiry(image_paths, prompt_texts, vlm_name='llavamed', **kwargs):
    #TODO
    answer = None
    usage = 0
    return answer, usage









# image_paths = ['/home/ivan/Downloads/image1.jpeg','/home/ivan/Downloads/image2.jpeg']
# prompt_texts = ["What is shown on the first image? What is the name of the file of the first image? What is the size of the image in pixels?","What is shown on the second image? What is the name of the file of the second image? What is the size of the image in pixels?"]
    
# answer, usage = gemini_multiimage_api_visual_inquiry(image_paths, prompt_texts)


# image_path = '/home/ivan/Helmholtz/VLMevaluation/Datasets/AML_Matek_structured/image_1.png'
# prompt_text = "What's in the image?"
#  answer, usage = gemini_api_visual_inquiry(image_path, prompt_text)


# prompt_text = "What day is today?"
# answer, usage = gemini_api_text_inquiry(prompt_text)
# print(answer)
# print(usage.total_token_count)
