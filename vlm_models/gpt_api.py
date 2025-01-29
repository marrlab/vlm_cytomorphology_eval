#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:50:09 2024

@author: ivan
"""


import base64
from openai import OpenAI
# import os
# import time
# import pandas as pd
import json

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def gpt_api_visual_inquiry(image_path, prompt_text, vlm_name='gpt-4o', **kwargs):
    # Getting the base64 string
    base64_image = encode_image(image_path)
    
    # Set default detail if not provided in kwargs
    detail = kwargs.get('detail', 'low')
    
    response = client.chat.completions.create(
      model=vlm_name, #"gpt-4o-mini"
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt_text,
            },
            {
              "type": "image_url",
              "image_url": {
                "url":  f"data:image/jpeg;base64,{base64_image}",
                "detail": detail
              },
            },
          ],
        }
      ],
    )
    
    answer=response.choices[0].message.content
    
    
    # Extract and print token usage
    usage = response.usage.total_tokens
    
    return answer, usage


def gpt_api_text_inquiry(prompt_text, vlm_name='gpt-4o', **kwargs):
    response = client.chat.completions.create(
        model=vlm_name, #"o1-preview",
        messages=[
            {
                "role": "user", 
                "content": prompt_text
            }
        ]
    )

    answer=response.choices[0].message.content
    
    
    # Extract and print token usage
    usage = response.usage.total_tokens
    
    return answer, usage


def gpt_api_finetuning_entry(prompt_text, image_url, ground_truth_label):

    prompt_text = json.dumps(prompt_text)

    entry = '{"messages": [{"role": "system", "content": "You are an assistant that identifies call types."}, {"role": "user", "content": ' + prompt_text + '}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "' + image_url + '"}}]}, {"role": "assistant", "content": "' + ground_truth_label + '"}]}'

    # jsonl_entry = json.dumps(entry)

    return entry


def gpt_api_finetune(train_fine_tuning_jsonl_path, val_fine_tuning_jsonl_path, n_epochs):


    client.fine_tuning.jobs.create(
        training_file=train_fine_tuning_jsonl_path,
        validation_file=val_fine_tuning_jsonl_path,
        model="gpt-4o-mini",
        method={
    "type": "supervised",
    "supervised": {
      "hyperparameters": {
                "n_epochs": n_epochs
                }
            }
        }
    )

