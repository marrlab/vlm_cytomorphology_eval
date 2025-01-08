#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:17:37 2024

@author: ivan
"""


import PIL.Image
import os
import google.generativeai as genai




def gemini_api_visual_inquiry(image_path, prompt_text, vlm_name='gemini-2.0-flash-exp', **kwargs): #'gemini-1.5-pro'

    image = PIL.Image.open(image_path)

    #Choose a Gemini model.
    model = genai.GenerativeModel(model_name=vlm_name)

    response = model.generate_content([prompt_text, image])

    answer = response.text
    usage = response.usage_metadata.total_token_count
    
    return answer, usage


def gemini_api_text_inquiry(prompt_text, vlm_name='gemini-2.0-flash-exp', **kwargs): #'gemini-1.5-pro'

    model = genai.GenerativeModel(model_name=vlm_name)
    
    response = model.generate_content(prompt_text)

    answer = response.text

    usage = response.usage_metadata.total_token_count
    
    return answer, usage



# image_path = '/home/ivan/Helmholtz/VLMevaluation/Datasets/AML_Matek_structured/image_1.png'
# prompt_text = "What's in the image?"
#  answer, usage = gemini_api_visual_inquiry(image_path, prompt_text)


# prompt_text = "What day is today?"
# answer, usage = gemini_api_text_inquiry(prompt_text)
# print(answer)
# print(usage.total_token_count)
