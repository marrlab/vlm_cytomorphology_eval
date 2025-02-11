#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:31:31 2025

@author: ivan
"""

from dataset_info_and_paths import get_global_info, get_dataset_info, get_result_path
import pandas as pd
import os

if __name__ == "__main__":
    global_info = get_global_info()

    available_datasets = global_info['available_datasets']
    available_task_types = global_info['available_task_types']
    # available_model_families = global_info['available_model_families']
    available_models = global_info['available_models']
    # recommended_models = global_info['recommended_models']
    available_task_types.remove('nonstructured')

    for reviewed in [False, True]:
        if reviewed == False:
            print('*Results computed with VLM models:*')
        else:
            print('*Answers reviewed:*')

        for dataset_name in available_datasets:
            print(f'*Results for dataset: {dataset_name}:*')

            if dataset_name == 'HiCervix':
                continue

            dataset_info = get_dataset_info(dataset_name, dataset_type = 'test')
            vlm_eval_subset_labels_path = dataset_info['vlm_eval_subset_labels_path']

            # Load the subset labels CSV file
            subset_labels_df = pd.read_csv(vlm_eval_subset_labels_path)

            len_dataset = len(subset_labels_df)

            for task_type in available_task_types:
                print(f'*Results for task type: {task_type}:*')

                results_fully_computed = []
                results_partially_computed = []
                results_not_computed = []

                for vlm_name in available_models:  
                    if ('acevedo' in vlm_name) and ('Acevedo' not in dataset_name):
                        continue

                    if ('acevedo' in vlm_name) and (task_type == '1shot_classification'):
                        continue

                    result_paths = get_result_path(vlm_name, dataset_name, task_type, reviewed, file_type_extension='csv')
                    answers_path = result_paths['answers_path']

                    if os.path.exists(answers_path):
                        answers_df = pd.read_csv(answers_path)
                        len_answers = len(answers_df)

                        if len_answers == len_dataset:
                            results_fully_computed.append(vlm_name)
                        else:
                            results_partially_computed.append(vlm_name+f' ({len_answers}/{len_dataset})')
                    else:
                        results_not_computed.append(vlm_name) # + f' {answers_path}')

                print('*Fully computed:*')
                for model in results_fully_computed:
                    print(f'\t{model}')
                
                print('*Partially computed:*')
                for model in results_partially_computed:
                    print(f'\t{model}')
                
                print('*Not computed:*')
                for model in results_not_computed:
                    print(f'\t{model}')

            




    
    

