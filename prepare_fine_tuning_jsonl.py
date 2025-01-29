#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:28:52 2025

@author: ivan
"""

from prompts import get_prompt
import pandas as pd
from dataset_info_and_paths import get_dataset_info, get_fine_tuning_jsonl_path


def import_model(model_family):
    if 'gpt' in model_family:
        from vlm_models.gpt_api import gpt_api_finetuning_entry
        api_finetuning_entry = gpt_api_finetuning_entry
    else:
        raise ValueError(f"Model family {model_family} not found")
        return None

    return api_finetuning_entry


def prepare_fine_tuning_jsonl(n_train_samples_per_label, dataset_name, task_type, model_family):

    dataset_info = get_dataset_info(dataset_name)

    vlm_eval_subset_labels_path = dataset_info['vlm_eval_subset_labels_path']
    sorting_label_column_in_csv = dataset_info['sorting_label_column_in_csv']
    vlm_eval_subset_oline_location = dataset_info['vlm_eval_subset_oline_location']
    abbreviation_dict_path = dataset_info['abbreviation_dict_path']

    dataset_labels = pd.read_csv(vlm_eval_subset_labels_path)

    cells_to_include = pd.DataFrame(columns=dataset_labels.columns)

    prompt_text = list(get_prompt(dataset_name, task_type, reviewed=False).values())[0]

    api_finetuning_entry = import_model(model_family)

    fine_tuning_jsonl_path = get_fine_tuning_jsonl_path(n_train_samples_per_label, dataset_name, task_type, model_family)

    if abbreviation_dict_path is not None:
        abbreviation_dict = pd.read_csv(abbreviation_dict_path, usecols=lambda x: x.strip())
        abbreviation_dict = {row[0]: row[1].replace(' \t','') for _, row in abbreviation_dict.iterrows()}
        

    unique_labels = dataset_labels[sorting_label_column_in_csv].unique()

    for label in unique_labels:
        label_samples = dataset_labels[dataset_labels[sorting_label_column_in_csv] == label]

        n_label = min(n_train_samples_per_label, len(label_samples))

        if n_label < n_train_samples_per_label:
            print(f'Warning: {dataset_name} has only {n_label} samples for label {label}.')

        label_samples = label_samples.iloc[0:n_label]

        cells_to_include = pd.concat([cells_to_include, label_samples])

    cells_to_include = cells_to_include.sample(frac=1).reset_index(drop=True)

    # First empty the file if it exists
    open(fine_tuning_jsonl_path, 'w').close()
    
    # Now open for appending
    with open(fine_tuning_jsonl_path, 'a') as f:
        for _, row in cells_to_include.iterrows():
            image_name = row['image_name']
            image_url = vlm_eval_subset_oline_location + image_name + '.jpg'
            ground_truth_label = row[sorting_label_column_in_csv]
            
            if abbreviation_dict_path is not None:
                label_text = ground_truth_label + ' ' + abbreviation_dict[ground_truth_label]
            else:
                label_text = ground_truth_label

            jsonl_entry = api_finetuning_entry(prompt_text, image_url, label_text)
            f.write(jsonl_entry + '\n')



    return 0

        


dataset_name='Bone_Marrow_Cyto_train'
task_type='0shot_classification'
model_family='gpt'

for n_train_samples_per_label in [1, 5, 10, 25, 50]:
    prepare_fine_tuning_jsonl(n_train_samples_per_label, dataset_name, task_type, model_family)

dataset_name='Bone_Marrow_Cyto_val'
task_type='0shot_classification'
model_family='gpt'

for n_train_samples_per_label in [1, 5, 10]:
    prepare_fine_tuning_jsonl(n_train_samples_per_label, dataset_name, task_type, model_family)