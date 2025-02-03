#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:12:37 2025

@author: ivan
"""



def remove_image_from_finetuning_jsonl(line_number, jsonl_path):
    # Open and read the jsonl file
    with open(jsonl_path, 'r') as f:
        # Read all lines
        lines = f.readlines()
        
        # Since line_number is 1-based, subtract 1 to get 0-based index
        index = line_number - 1
        
        if index < 0 or index >= len(lines):
            print(f"Error: Line number {line_number} is out of range. File has {len(lines)} lines.")
            return
            
        # Print the line at the specified line number
        print(f"Removing line {line_number}:")
        print(lines[index])

        # Remove the line from the file
        lines.pop(index)

        # Write the updated lines back to the file
        output_path = jsonl_path.rsplit('.', 1)[0] + '_line_removed.' + jsonl_path.rsplit('.', 1)[1]
        with open(output_path, 'w') as f:
            f.writelines(lines)



# line_number = 618
# jsonl_path = '/home/ivan/Helmholtz/VLMevaluation/Datasets/Acevedo/train/fine_tuning_Acevedo_train_task_type_0shot_classification_model_family_gpt_n_per_label_100.jsonl'

# remove_image_from_finetuning_jsonl(line_number, jsonl_path)