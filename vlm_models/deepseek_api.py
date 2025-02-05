#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:17:37 2024

@author: ivan
"""

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

def load_deepseek_model(vlm_name='deepseek-vl2-small'):


    if vlm_name == 'deepseek-vl2-tiny':
        model_path = os.path.join("deepseek-ai/deepseek-vl2-tiny")
    elif vlm_name == 'deepseek-vl2-small':
        model_path = os.path.join("deepseek-ai/deepseek-vl2-small")
    elif vlm_name == 'deepseek-vl2':
        model_path = os.path.join("deepseek-ai/deepseek-vl2")
    else:
        raise ValueError(f"Model {vlm_name} not supported")

    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
   
    return vl_gpt, tokenizer



def deepseek_api_visual_inquiry(image_path, prompt_text, vlm_name='deepseek-vl2-small', **kwargs): #'gemini-1.5-pro'

    vl_gpt = kwargs.get('vl_gpt')
    tokenizer = kwargs.get('tokenizer')
    max_new_tokens = kwargs.get('max_new_tokens', 10000) # 512
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Please note that <|ref|> and <|/ref|> are designed specifically for the object localization feature. These special tokens are not required for normal conversations.
    ## If you would like to experience the grounded captioning functionality (responses that include both object localization and reasoning), you need to add the special token <|grounding|> at the beginning of the prompt. Examples could be found in Figure 9 of our paper.
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n<|ref|>" + prompt_text + "<|/ref|>.",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)

    usage = 0 

    return answer, usage

def deepseek_api_text_inquiry(prompt_text, vlm_name='deepseek-vl2-small', **kwargs):

    vl_gpt = kwargs.get('vl_gpt')
    tokenizer = kwargs.get('tokenizer')
    max_new_tokens = kwargs.get('max_new_tokens', 10000) # 512
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Please note that <|ref|> and <|/ref|> are designed specifically for the object localization feature. These special tokens are not required for normal conversations.
    ## If you would like to experience the grounded captioning functionality (responses that include both object localization and reasoning), you need to add the special token <|grounding|> at the beginning of the prompt. Examples could be found in Figure 9 of our paper.
    conversation = [
        {
            "role": "<|User|>",
            "content": "<|ref|>" + prompt_text + "<|/ref|>.",
            "images": [],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)

    usage = 0 

    return answer, usage


def deepseek_multiimage_api_visual_inquiry(image_paths, prompt_texts, vlm_name='deepseek-vl2-small', **kwargs): #'gemini-1.5-pro'

    if len(image_paths) != len(prompt_texts):
        raise ValueError("The number of image paths and prompt texts must be the same.")

    def prepare_conversation(prompt_texts, image_paths):

        ## Please note that <|ref|> and <|/ref|> are designed specifically for the object localization feature. These special tokens are not required for normal conversations.
        ## If you would like to experience the grounded captioning functionality (responses that include both object localization and reasoning), you need to add the special token <|grounding|> at the beginning of the prompt. Examples could be found in Figure 9 of our paper.
        # multiple images/interleaved image-text
        conversation = [
            {
                "role": "<|User|>",
                "content": "",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]
            
        for prompt_text, image_path in zip(prompt_texts, image_paths):
            
            conversation[0]["content"] = conversation[0]["content"] + '<image>' + prompt_text + '\n'
            conversation[0]["images"].append(image_path)
        
        return conversation

    vl_gpt = kwargs.get('vl_gpt')
    tokenizer = kwargs.get('tokenizer')
    max_new_tokens = kwargs.get('max_new_tokens', 10000) # 512
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conversation = prepare_conversation(prompt_texts, image_paths)   

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)

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
