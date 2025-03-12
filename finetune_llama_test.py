#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:08:57 2025

@author: ivan
"""

from unsloth import FastVisionModel
import torch
from transformers import  AutoProcessor



def FV_load_llama_model(vlm_name='llama-3.2-multimodal-11B'):
    if vlm_name == 'llama-3.2-multimodal-11B':
        model_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/llama-3.2-multimodal-11B/llama-3.2-11B-model"
        processor_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/llama-3.2-multimodal-11B/llama-3.2-11B-processor"

    elif vlm_name == 'llama-3.2-multimodal-90B':
        model_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/llama-3.2-multimodal-90B/llama-3.2-90B-model"
        processor_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/llama-3.2-multimodal-90B/llama-3.2-90B-processor"
    else:
        raise ValueError(f"Model {vlm_name} not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # model, tokenizer = FastVisionModel.from_pretrained(
    #     model_path,
    #     load_in_4bit=True,
    #     use_gradient_checkpointing="unsloth",
    # )
    
    # print('Successfully loaded the model: '+vlm_name)
    
    # print(model)
    # print(tokenizer)
    
    # return model, tokenizer    
    
    
    # Load model
    model, _ = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    
    processor = AutoProcessor.from_pretrained(processor_path)
    
    
    print('Successfully loaded the model: '+vlm_name)
    
    print(model)
    print(processor)    
    
    
    return model, processor #tokenizer

FV_load_llama_model()