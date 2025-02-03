#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:28:52 2025

@author: ivan
"""

from prompts import get_prompt
import pandas as pd
import numpy as np
import os
import shutil
from dataset_info_and_paths import get_global_info, get_dataset_info, get_fine_tuning_subset_paths


def import_model(model_family):
    if 'gpt' in model_family:
        from vlm_models.gpt_api import gpt_api_finetuning_entry
        api_finetuning_entry = gpt_api_finetuning_entry
    else:
        raise ValueError(f"Model family {model_family} not found")
        return None

    return api_finetuning_entry

def prepare_github_upload_folders(dataset_name, dataset_type):
    """
    Prepare folders for github upload.
    """

    global_info = get_global_info()

    github_upload_root_folder_path = global_info['github_upload_root_folder_path']

    os.makedirs(github_upload_root_folder_path, exist_ok=True)

    dataset_info = get_dataset_info(dataset_name, dataset_type)

    vlm_eval_subset_folder_path = dataset_info['vlm_eval_subset_folder_path']

    # List all image files in vlm_eval_subset_folder_path
    image_files = []
    for f in os.listdir(vlm_eval_subset_folder_path):
        if f.startswith('image_') and any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            image_files.append(f)
    
    # Sort by the number in the filename
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    n_folders = int(np.ceil(len(image_files) / 1000))

    # Create folders and copy files in batches of 1000
    for i in range(n_folders):
        batch_num = i + 1
        batch_folder = os.path.join(github_upload_root_folder_path, 
                                  f"{dataset_name}_{dataset_type}_{batch_num}")
        os.makedirs(batch_folder, exist_ok=True)
        
        # Get files for this batch
        batch_files = image_files[i*1000:(i+1)*1000]
        
        # Copy each file to the batch folder
        for img_file in batch_files:
            src = os.path.join(vlm_eval_subset_folder_path, img_file)
            dst = os.path.join(batch_folder, img_file)
            shutil.copy2(src, dst)



def prepare_fine_tuning_jsonl_and_csv(n_train_samples_per_label, dataset_name, dataset_type, task_type, model_family):

    dataset_info = get_dataset_info(dataset_name, dataset_type)

    vlm_eval_subset_labels_path = dataset_info['vlm_eval_subset_labels_path']
    # sampling_label_column_in_csv = dataset_info['sampling_label_column_in_csv']
    vlm_eval_subset_oline_locations = dataset_info['vlm_eval_subset_oline_locations']
    abbreviation_dict_path = dataset_info['abbreviation_dict_path']

    dataset_labels = pd.read_csv(vlm_eval_subset_labels_path)

    cells_to_include = pd.DataFrame(columns=dataset_labels.columns)

    prompt_text = list(get_prompt(dataset_name, task_type, reviewed=False).values())[0]

    api_finetuning_entry = import_model(model_family)

    finetuning_paths = get_fine_tuning_subset_paths(n_train_samples_per_label, dataset_name, dataset_type, task_type, model_family)
    fine_tuning_jsonl_path = finetuning_paths['fine_tuning_jsonl_path']
    fine_tuning_csv_path = finetuning_paths['fine_tuning_csv_path']

    if abbreviation_dict_path is not None:
        abbreviation_dict = pd.read_csv(abbreviation_dict_path, usecols=lambda x: x.strip())
        abbreviation_dict = {row[0]: row[1].replace(' \t','') for _, row in abbreviation_dict.iterrows()}
        

    unique_labels = dataset_labels['label'].unique() #sampling_label_column_in_csv

    for label in unique_labels:
        label_samples = dataset_labels[dataset_labels['label'] == label] #sampling_label_column_in_csv

        n_label = min(n_train_samples_per_label, len(label_samples))

        if n_label < n_train_samples_per_label:
            print(f'Warning: {dataset_name} has only {n_label} samples for label {label}.')

        label_samples = label_samples.iloc[0:n_label]

        cells_to_include = pd.concat([cells_to_include, label_samples])

    cells_to_include = cells_to_include.sample(frac=1).reset_index(drop=True)

    cells_to_include.to_csv(fine_tuning_csv_path, index=False)

    # First empty the file if it exists
    open(fine_tuning_jsonl_path, 'w').close()
    
    # Now open for appending
    with open(fine_tuning_jsonl_path, 'a') as f:
        for _, row in cells_to_include.iterrows():
            image_name = row['image_name']
            if isinstance(vlm_eval_subset_oline_locations, list):
                image_number = int(image_name.split('_')[-1])
                location_index = image_number // 1000
                image_url = vlm_eval_subset_oline_locations[location_index] + image_name + '.jpg'
            else:
                image_url = vlm_eval_subset_oline_locations + image_name + '.jpg'
            ground_truth_label = row['label'] #sampling_label_column_in_csv
            
            if abbreviation_dict_path is not None:
                label_text = ground_truth_label + ' ' + abbreviation_dict[ground_truth_label]
            else:
                label_text = ground_truth_label

            jsonl_entry = api_finetuning_entry(prompt_text, image_url, label_text)
            f.write(jsonl_entry + '\n')



    return 0

        
# dataset_name='Acevedo'
# dataset_type='train'
# prepare_github_upload_folders(dataset_name, dataset_type)

# dataset_type = 'val'
# prepare_github_upload_folders(dataset_name, dataset_type)

dataset_name='Acevedo'
task_type='0shot_classification'
model_family='gpt'

dataset_type = 'train'

for n_train_samples_per_label in [90]: #[1, 5, 10, 25, 50,100,200]:
    prepare_fine_tuning_jsonl_and_csv(n_train_samples_per_label, dataset_name, dataset_type, task_type, model_family)

# dataset_type = 'val'

# n_train_samples_per_label = 50

# prepare_fine_tuning_jsonl_and_csv(n_train_samples_per_label, dataset_name, dataset_type, task_type, model_family)