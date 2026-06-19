#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:17:37 2024

@author: ivan
"""


import PIL.Image
import os
import google.generativeai as genai

_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if _api_key:
    genai.configure(api_key=_api_key)




def gemini_api_visual_inquiry(image_path, prompt_text, vlm_name='gemini-2.0-flash-exp', **kwargs): #'gemini-1.5-pro'

    image = PIL.Image.open(image_path)

    #Choose a Gemini model.
    model = genai.GenerativeModel(model_name=vlm_name)

    response = model.generate_content([prompt_text, image])

    answer = response.text
    usage = response.usage_metadata.total_token_count
    
    return answer, usage

def gemini_multiimage_api_visual_inquiry(image_paths, prompt_texts, vlm_name='gemini-2.0-flash-exp', **kwargs): #'gemini-1.5-pro'

    if len(image_paths) != len(prompt_texts):
        raise ValueError("The number of image paths and prompt texts must be the same.")

    messages = []

    for image_path, prompt_text in zip(image_paths, prompt_texts):  
        image = PIL.Image.open(image_path)

        messages.append(image)
        messages.append(prompt_text)
        
    #Choose a Gemini model.
    model = genai.GenerativeModel(model_name=vlm_name)

    response = model.generate_content(messages)

    answer = response.text
    usage = response.usage_metadata.total_token_count
    
    return answer, usage

def gemini_api_text_inquiry(prompt_text, vlm_name='gemini-2.0-flash-exp', **kwargs): #'gemini-1.5-pro'

    model = genai.GenerativeModel(model_name=vlm_name)
    
    response = model.generate_content(prompt_text)

    answer = response.text

    usage = response.usage_metadata.total_token_count
    
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
