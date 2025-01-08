#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:29:35 2024

@author: ivan
"""

from dataset_info_and_paths import get_dataset_info, get_result_path, get_review_model
from prompts import get_prompt
import re
import os
import time
import pandas as pd



def run_api_inquiry(vlm_name, dataset_name, structured_nonstructured):
    """
    Run visual language model API inquiries on a dataset of images.
    
    Args:
        vlm_name (str): Name of the visual language model to use (e.g. 'gpt-o', 'gemini')
        dataset_name (str): Name of the dataset to evaluate ('AML_Matek', 'Bone_Marrow_Cyto', 'WBCAtt') 
        structured_nonstructured (str): Whether to use structured or nonstructured prompts ('structured', 'nonstructured')
        
    Returns:
        tuple: A tuple containing:
            - answers_df (pd.DataFrame): DataFrame with image names and model answers for each category
            - total_tokens_used_df (pd.DataFrame): DataFrame with image names and token usage for each category
    """

    structured_nonstructured_review = structured_nonstructured
        

    dataset_info = get_dataset_info(dataset_name)

    vlm_eval_subset_folder_path = dataset_info['vlm_eval_subset_folder_path']

    prompt = get_prompt(dataset_name, structured_nonstructured_review)

    categories = list(prompt.keys())

    if 'gpt' in vlm_name:
        from vlm_models.gpt_api import gpt_api_visual_inquiry
        kwargs = {'detail': 'low'}
        api_visual_inquiry = gpt_api_visual_inquiry
        sleep_time=1
    elif 'gemini' in vlm_name:
        from vlm_models.gemini_api import gemini_api_visual_inquiry
        kwargs = {}
        api_visual_inquiry = gemini_api_visual_inquiry
        sleep_time=1
    elif 'llama' in vlm_name:
        from vlm_models.llama_api import load_llama_model, llama_api_visual_inquiry
        model, processor = load_llama_model(vlm_name)
        kwargs = {'model': model, 'processor': processor, 'max_new_tokens': 10000}
        api_visual_inquiry = llama_api_visual_inquiry
        sleep_time=0.2

 
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
    
    answers_path_base = get_result_path(vlm_name, dataset_name, structured_nonstructured, 'answers', None)
    usage_path_base = get_result_path(vlm_name, dataset_name, structured_nonstructured, 'total_tokens_used', None)

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

def run_api_review(vlm_name: str, dataset_name: str, **kwargs):
    """
    Run API review on previously generated answers.
    
    Args:
        vlm_name (str): Name of the VLM model that was used to generate the answers
        dataset_name (str): Name of the dataset to evaluate
        **kwargs: Additional arguments to pass to api_visual_inquiry
        
    Returns:
        tuple: (reviewed_answers_df, total_tokens_used_review_df) containing the reviewed answers and token usage
    """


    # structured_yes_no = 'yes'
    structured_nonstructured = 'structured'

    review_model = get_review_model(vlm_name)
    
    # Get folder path based on cluster/local setting
    # dataset_info = get_dataset_info(dataset_name, structured_yes_no)
    # vlm_eval_subset_folder_path = dataset_info['vlm_eval_subset_folder_path']

    if 'gpt' in review_model:
        from vlm_models.gpt_api import gpt_api_text_inquiry
        kwargs = {}
        api_text_inquiry = gpt_api_text_inquiry
        sleep_time=1
    elif 'gemini' in review_model:
        from vlm_models.gemini_api import gemini_api_text_inquiry
        kwargs = {}
        api_text_inquiry = gemini_api_text_inquiry
        sleep_time=1
    elif 'llama' in review_model:
        from vlm_models.llama_api import load_llama_model, llama_api_text_inquiry  
        model, processor = load_llama_model(review_model)
        kwargs = {'model': model, 'processor': processor, 'max_new_tokens': 20000}
        api_text_inquiry = llama_api_text_inquiry
        sleep_time=0.2

    # Get prompt dictionary for review
    prompt = get_prompt(dataset_name, 'review')
    
    # Load original answers
    answers_path = get_result_path(vlm_name, dataset_name, structured_nonstructured, 'answers', 'csv')
    original_answers_df = pd.read_csv(answers_path)
    
    # Create dataframes to store results
    reviewed_answers_df = pd.DataFrame(index=original_answers_df.index, columns=original_answers_df.columns)
    total_tokens_used_review_df = pd.DataFrame(index=original_answers_df.index, columns=original_answers_df.columns)

    # Get categories from prompt dictionary
    categories = [col for col in list(prompt.keys()) if col != 'image_name']
    
    answers_path_base = get_result_path(vlm_name, dataset_name, 'reviewed', 'answers', None)
    usage_path_base = get_result_path(vlm_name, dataset_name, 'reviewed', 'total_tokens_used', None)

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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run VLM model evaluation on cytomorphology datasets')
    parser.add_argument('--vlm_name', type=str, choices=['gpt-4o', 'gemini-2.0-flash-exp', 'llama-3.2-multimodal-11B'],
                      help='Name of VLM model to use.')
    parser.add_argument('--dataset_name', type=str, choices=['AML_Matek', 'Bone_Marrow_Cyto', 'WBCAtt'],
                      help='Name of dataset to evaluate.')
    parser.add_argument('--structured_nonstructured', type=str, choices=['structured', 'nonstructured'],
                      help='Whether to use structured or nonstructured prompts')
    parser.add_argument('--run_review', action='store_true',
                      help='Whether to also run review after inquiry')

    args = parser.parse_args()
    
    # Run the API inquiry
    run_api_inquiry(args.vlm_name, args.dataset_name, args.structured_nonstructured)
    
    # Optionally run review
    if args.run_review:
        run_api_review(args.vlm_name, args.dataset_name)
