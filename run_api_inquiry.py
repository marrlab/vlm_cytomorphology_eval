#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:29:35 2024

@author: ivan
"""

from dataset_info_and_paths import get_dataset_info, get_result_path, get_review_model, get_global_info 
from prompts import get_prompt
import re
import os
import time
import pandas as pd


def import_model(vlm_name):
    if 'gpt' in vlm_name:
        from vlm_models.gpt_api import gpt_api_visual_inquiry, gpt_api_text_inquiry
        kwargs = {'detail': 'low'}
        api_visual_inquiry = gpt_api_visual_inquiry
        api_text_inquiry = gpt_api_text_inquiry
        sleep_time=1
    elif 'gemini' in vlm_name:
        from vlm_models.gemini_api import gemini_api_visual_inquiry, gemini_api_text_inquiry
        kwargs = {}
        api_visual_inquiry = gemini_api_visual_inquiry
        api_text_inquiry = gemini_api_text_inquiry
        sleep_time=1
    elif 'llama' in vlm_name:
        from vlm_models.llama_api import load_llama_model, llama_api_visual_inquiry, llama_api_text_inquiry
        model, processor = load_llama_model(vlm_name)
        kwargs = {'model': model, 'processor': processor, 'max_new_tokens': 20000}
        api_visual_inquiry = llama_api_visual_inquiry
        api_text_inquiry = llama_api_text_inquiry
        sleep_time=0.2
    else:
        raise ValueError(f"Model {vlm_name} not found")
        return None, None, None, None

    return api_visual_inquiry, api_text_inquiry, kwargs, sleep_time


def run_api_0shot_classification(vlm_name, dataset_name):
    """
    Run visual language model API inquiries on a dataset of images.
    
    Args:
        vlm_name (str): Name of the visual language model to use (see get_global_info()['recommended_models'])
        dataset_name (str): Name of the dataset to evaluate (see get_global_info()['available_datasets'])
        
    Returns:
        tuple: A tuple containing:
            - answers_df (pd.DataFrame): DataFrame with image names and model answers for each category
            - total_tokens_used_df (pd.DataFrame): DataFrame with image names and token usage for each category
    """

    api_visual_inquiry, api_text_inquiry, kwargs, sleep_time = import_model(vlm_name) 

    dataset_info = get_dataset_info(dataset_name, 'test')

    vlm_eval_subset_folder_path = dataset_info['vlm_eval_subset_folder_path']

    prompt = get_prompt(dataset_name, '0shot_classification', reviewed=False)

    categories = [col for col in list(prompt.keys()) if col != 'image_name']   
 
    # Get list of image files
    # Extract number from image name if it exists, otherwise use the full name
    def get_sort_key(filename):        
        match = re.search(r'image_(\d+)', filename)
        return int(match.group(1)) if match else filename
    
    image_files = sorted([f for f in os.listdir(vlm_eval_subset_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], 
                        key=get_sort_key)

    image_names = [os.path.splitext(image_file)[0] for image_file in image_files]

    # Create dataframes to store results
    answers_df = pd.DataFrame(columns=['image_name', *categories])
    total_tokens_used_df = pd.DataFrame(columns=['image_name', *categories])
    
    answers_path_base = get_result_path(vlm_name, dataset_name, '0shot_classification', reviewed=False, file_type_extension=None)['answers_path']
    usage_path_base = get_result_path(vlm_name, dataset_name, '0shot_classification', reviewed=False, file_type_extension=None)['tokens_path']

    # Process each image
    for j, category in enumerate(categories):
        # Get prompt text for this category
        prompt_text = prompt[category]
        
        for i, image_file in enumerate(image_files):
            # Print progress every 25 images
            if i % 25 == 0:
                print(f"Evaluating {dataset_name} with {vlm_name}: category {j+1}/{len(categories)}, image {i+1}/{len(image_files)}")
            
            # Get image name without extension
            image_name = image_names[i]
            
            # Full path to image
            image_path = os.path.join(vlm_eval_subset_folder_path, image_file)
            
            # Run API inquiry
            answer, usage = api_visual_inquiry(image_path, prompt_text, vlm_name=vlm_name, **kwargs)
            
       
            answers_df.loc[i, 'image_name'] = image_name
            answers_df.loc[i, category] = answer
            total_tokens_used_df.loc[i, 'image_name'] = image_name
            total_tokens_used_df.loc[i, category] = usage
            
            # Sleep to avoid rate limit
            time.sleep(sleep_time)

            # Save every 25 images
            if (i + 1) % 25 == 0 or i == len(image_files) - 1:
                # Save answers dataframe
                answers_df.to_csv(answers_path_base + '.csv', index=False)
                answers_df.to_excel(answers_path_base + '.xlsx', index=False)
                
                # Save usage dataframe
                total_tokens_used_df.to_csv(usage_path_base + '.csv', index=False)
                total_tokens_used_df.to_excel(usage_path_base + '.xlsx', index=False)
    
    return answers_df, total_tokens_used_df

def run_api_review(vlm_name: str, dataset_name: str, task_type: str, **kwargs):
    """
    Run API review on previously generated answers.
    
    Args:
        vlm_name (str): Name of the VLM model that was used to generate the answers
        dataset_name (str): Name of the dataset to evaluate (see get_global_info()['available_datasets'])
        task_type (str): Type of task (see get_global_info()['available_task_types'])
        **kwargs: Additional arguments to pass to api_visual_inquiry
        
    Returns:
        tuple: (reviewed_answers_df, total_tokens_used_review_df) containing the reviewed answers and token usage
    """

    review_model = get_review_model(vlm_name)
    
    api_visual_inquiry, api_text_inquiry, kwargs, sleep_time = import_model(review_model)  

    # Get prompt dictionary for review
    prompt = get_prompt(dataset_name, task_type, reviewed=True)
    
    # Load original answers
    answers_path = get_result_path(vlm_name, dataset_name, task_type, reviewed=False, file_type_extension='csv')['answers_path']
    original_answers_df = pd.read_csv(answers_path)
    
    # Create dataframes to store results
    reviewed_answers_df = pd.DataFrame(index=original_answers_df.index, columns=original_answers_df.columns)
    total_tokens_used_review_df = pd.DataFrame(index=original_answers_df.index, columns=original_answers_df.columns)

    # Get categories from prompt dictionary
    categories = [col for col in list(prompt.keys()) if col != 'image_name']
    
    answers_path_base = get_result_path(vlm_name, dataset_name, task_type, reviewed=True, file_type_extension='csv')['answers_path']
    usage_path_base = get_result_path(vlm_name, dataset_name, task_type, reviewed=True, file_type_extension='csv')['tokens_path']

    # Process each image
    for j, category in enumerate(categories):
        # Get review prompt text for this category
        prompt_text = prompt[category]
        
        if 'llama' in review_model:
            prompt_text = prompt_text + "\n Please write a short answer (only the chosen class and nothing else)!"
        
        for i, row in original_answers_df.iterrows():
            # Print progress every 25 images
            if i % 25 == 0:
                print(f"Reviewing with {review_model} the {dataset_name} answers generated by {vlm_name}: category {j+1}/{len(categories)}, image {i+1}/{len(original_answers_df)}")
            
            # Get image name from original answers dataframe
            image_name = row['image_name']
                        
            # Get original answer for this category
            original_answer = row[category]
            
            # Create review prompt by combining original answer with review prompt
            review_prompt = f"Chatbot answered: {original_answer}\n{prompt_text}"
            
            # Run API inquiry
            answer, usage = api_text_inquiry(review_prompt, vlm_name=review_model, **kwargs)
            
            # Store results
            reviewed_answers_df.loc[i, 'image_name'] = image_name
            reviewed_answers_df.loc[i, category] = answer
            total_tokens_used_review_df.loc[i, 'image_name'] = image_name
            total_tokens_used_review_df.loc[i, category] = usage
            
            # Sleep to avoid rate limit
            time.sleep(sleep_time)
            
            # Save every 25 images
            if (i + 1) % 25 == 0 or i == len(original_answers_df) - 1:
                # Save reviewed answers dataframe
                reviewed_answers_df.to_csv(answers_path_base + '.csv', index=False)
                reviewed_answers_df.to_excel(answers_path_base + '.xlsx', index=False)
                
                # Save usage dataframe
                total_tokens_used_review_df.to_csv(usage_path_base + '.csv', index=False)
                total_tokens_used_review_df.to_excel(usage_path_base + '.xlsx', index=False)
    
    return reviewed_answers_df, total_tokens_used_review_df

# dataset_name = 'AML_Matek'
# vlm_name = 'gpt-4o' # 'blabla'
# task_type = '0shot_classification'
# # run_api_review(vlm_name, dataset_name, task_type)
# run_api_0shot_classification(vlm_name, dataset_name)


# vlm_name = 'ft:gpt-4o-2024-08-06:personal:bonemarrow-n-pl-5:AuoO9vo2'
# dataset_name = 'Bone_Marrow_Cyto'

# run_api_0shot_classification(vlm_name, dataset_name)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run VLM model evaluation on cytomorphology datasets')
    parser.add_argument('--vlm_name', type=str, choices=get_global_info()['available_models'],
                      help='Name of VLM model to use.')
    parser.add_argument('--dataset_name', type=str, choices=get_global_info()['available_datasets'],
                      help='Name of dataset to evaluate.')
    parser.add_argument('--task_type', type=str, choices=get_global_info()['available_task_types'],
                      help='Type of task to evaluate.')
    parser.add_argument('--run_review', type=int, default=0,
                      help='Whether to also run review after inquiry')

    args = parser.parse_args()

    if args.task_type == '0shot_classification':
        # Run the API inquiry
        run_api_0shot_classification(args.vlm_name, args.dataset_name)
        
    # Optionally run review
    if args.run_review:
        run_api_review(args.vlm_name, args.dataset_name, args.task_type)
