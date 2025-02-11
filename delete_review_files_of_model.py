#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:10:31 2025

@author: ivan
"""

from dataset_info_and_paths import get_global_info
import argparse
import os   


if __name__ == '__main__':    
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Delete review files for a specific VLM model')
    parser.add_argument('--vlm_name', type=str, help='Name of the VLM model whose review files should be deleted')
    
    # Parse arguments
    args = parser.parse_args()
    vlm_name = args.vlm_name

    if 'gpt' in vlm_name:
        print("GPT is used to review all other models. This would delete all reviewed files. Exiting.")
        exit()

    global_info = get_global_info()
    results_root_folder_path = global_info['results_root_folder_path']

    # Create OldResults subfolder if it doesn't exist
    old_results_folder = os.path.join(results_root_folder_path, 'OldResults')
    os.makedirs(old_results_folder, exist_ok=True)

    # Walk through results directory
    
    
    # List to store matching files
    review_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(results_root_folder_path):
        for file in files:
            # Check if file contains both vlm_name and '_reviewed_'
            if (vlm_name in file) and ('_reviewed_' in file):

                file_path = os.path.join(root, file)
                review_files.append(file_path)
                
                # Copy file to OldResults folder
                target_file_path = os.path.join(old_results_folder, file)
                os.rename(file_path, target_file_path)
                print(f"Moved {file} to OldResults folder")

                # Delete the original file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted {file}")

    print(f"Deleted {len(review_files)} review files for {vlm_name}")