#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:43:06 2024

@author: ivan
"""

import os
import pandas as pd
import numpy as np
import shutil
from PIL import Image
import argparse
from dataset_info_and_paths import get_dataset_info


def sample_data_subset(
    dataset_name: str
):
    """Creates a subset by sampling equally from each class.

    Args:
        dataset_name (str): Name of the dataset to create subset from

    The code can be easily modified to create multiple nonoverlapping subsets of the dataset by introducing
    another parameter output_folder_names and uncommenting suitable lines in the code. If dataset_to_avoid_overlap_with
    is not None, the code will avoid overlap with the dataset_to_avoid_overlap_with dataset.
    """

    dataset_info = get_dataset_info(dataset_name)   

    vlm_eval_subset_labels_path = dataset_info['vlm_eval_subset_labels_path']


    if os.path.exists(vlm_eval_subset_labels_path ):
        print(f"Warning: Sampled subset of the dataset {dataset_name} already exists (found labels file at {vlm_eval_subset_labels_path})!")
        print("Skipping subset preparation to avoid overwriting existing data. If you still want to create a new dataset, you must:\n - first get the permission from all the collaborators in the project\n - then manually delete the existing images and labels\n - then run the code again.")
        return 

    vlm_eval_subset_labels_path = vlm_eval_subset_labels_path.replace('.csv', '').replace('.xlsx', '')
    original_full_dataset_path = dataset_info['original_full_dataset_path']
    dataset_csv_path = dataset_info['dataset_csv_path']
    vlm_eval_subset_folder_path = dataset_info['vlm_eval_subset_folder_path']
    n_samples_per_label = dataset_info['n_samples_per_label']
    paths_column_in_csv = dataset_info['paths_column_in_csv']
    sorting_label_column_in_csv = dataset_info['sorting_label_column_in_csv']
    which_classes = dataset_info['which_classes']
    column_labels_to_keep = dataset_info['column_labels_to_keep']
    datasets_to_avoid_overlap_with = dataset_info['datasets_to_avoid_overlap_with']


    if datasets_to_avoid_overlap_with is not None:
        datasets_to_avoid_overlap_with_labels_df = []

        for dataset_to_avoid_overlap in datasets_to_avoid_overlap_with:
            dataset_to_avoid_overlap_info = get_dataset_info(dataset_to_avoid_overlap)
            dataset_to_avoid_overlap_labels_path = dataset_to_avoid_overlap_info['vlm_eval_subset_labels_path']

            if not os.path.exists(dataset_to_avoid_overlap_labels_path ):
                raise ValueError(f"Dataset to avoid overlap with {dataset_to_avoid_overlap} labels not found at {dataset_to_avoid_overlap_labels_path}. Run prepare_dataset.py for this dataset first.")

            dataset_to_avoid_overlap_labels_df = pd.read_csv(dataset_to_avoid_overlap_labels_path)

            datasets_to_avoid_overlap_with_labels_df.append(dataset_to_avoid_overlap_labels_df)


    # Check if labels path exists
    

    # Load the labels file
    # Determine file format and load accordingly
    file_extension = os.path.splitext(dataset_csv_path)[1].lower()
    if file_extension == '.csv':
        labels_df = pd.read_csv(dataset_csv_path)
    elif file_extension == '.xlsx' or file_extension == '.xls':
        labels_df = pd.read_excel(dataset_csv_path)
    elif file_extension == '.json':
        labels_df = pd.read_json(dataset_csv_path)
    elif file_extension == '.parquet':
        labels_df = pd.read_parquet(dataset_csv_path)
    elif file_extension == '.dat':
        labels_df = pd.read_csv(dataset_csv_path, sep='\s+', header=None)  # Assumes space/tab delimited
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    if datasets_to_avoid_overlap_with is not None:
        for dataset_to_avoid_overlap_labels_df in datasets_to_avoid_overlap_with_labels_df:
            labels_df = labels_df[~labels_df[paths_column_in_csv].isin(dataset_to_avoid_overlap_labels_df['original_image_path'])]
    
    no_folders = 1 # len(output_folder_names) # Uncomment if you want create multiple nonoverlapping subsets
    

    # Handle 'all' option for which_classes
    if which_classes == 'all':
        which_classes = labels_df[sorting_label_column_in_csv].unique()

    if column_labels_to_keep == 'all':
        column_labels_to_keep = [col for col in labels_df.columns if col != paths_column_in_csv]

    # Add dataset name prefix to each folder name
    output_folder_names = ['test_'+dataset_name] # [dataset_name + '_' + name for name in output_folder_names] # Uncomment if you want to create multiple nonoverlapping subsets
    
    # Create a dictionary to hold dataframes for each folder
    folder_dfs = {folder_name: pd.DataFrame(columns=["original_image_path"] + column_labels_to_keep) for folder_name in output_folder_names}

    # Iterate through each current_class
    for class_idx, current_class in enumerate(which_classes):
        print(f"Processing current_class: {current_class} - current_class {class_idx+1}/{len(which_classes)}")

        # Filter the dataframe for the current current_class
        df_class = labels_df[labels_df[sorting_label_column_in_csv] == current_class]

        
        
        # Shuffle the dataframe
        df_class = df_class.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Adjust n_samples_per_label if not enough samples available        
        samples_per_folder = n_samples_per_label
        if len(df_class) < n_samples_per_label * no_folders:
            print(f"Warning: Not enough samples for current_class {current_class}. Required: {n_samples_per_label * no_folders}, Available: {len(df_class)}")
            print(f"Adjusting to {samples_per_folder} samples per folder for current_class {current_class}")
            samples_per_folder = len(df_class) // no_folders
        
            
        # Split the data across folders
        for i, folder_name in enumerate(output_folder_names):
            start_idx = i * samples_per_folder
            end_idx = start_idx + samples_per_folder
            selected_rows = df_class.iloc[start_idx:end_idx]
            
            # Append to the folder's dataframe
            folder_dfs[folder_name] = pd.concat(
                [folder_dfs[folder_name], selected_rows[[paths_column_in_csv] + column_labels_to_keep].rename(columns={paths_column_in_csv: "original_image_path"})],
                ignore_index=True
            )
    
    # Process each folder
    for folder_name, folder_df in folder_dfs.items():
        # Shuffle rows
        folder_df = folder_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add image_name column as first column
        folder_df.insert(0, "image_name", ["image_" + str(i) for i in range(len(folder_df))])
        
        # Create folder in save_location
        folder_path = vlm_eval_subset_folder_path #os.path.join(vlm_eval_subset_folder_path, folder_name) # Uncomment if you want to create multiple nonoverlapping subsets
        os.makedirs(folder_path, exist_ok=True)
        
        print('Saving data subset to:', folder_path)

        # Copy and rename images
        for _, row in folder_df.iterrows():
            original_path = os.path.join(original_full_dataset_path, row["original_image_path"])
            file_ext = os.path.splitext(row["original_image_path"])[-1].lower()
            if file_ext not in ['.jpg', '.jpeg', '.png']:
                # Read and save as PNG if not already in desired format
                img = Image.open(original_path)
                new_path = os.path.join(folder_path, row["image_name"] + '.png')
                img.save(new_path, 'PNG')
            else:
                # Keep original format if already jpg/jpeg/png
                new_path = os.path.join(folder_path, row["image_name"] + file_ext)
                shutil.copy2(original_path, new_path)
        
        # Save the dataframe as CSV and XLS
        folder_df.to_csv(vlm_eval_subset_labels_path+'.csv', index=False) # Correct here if you want to create multiple nonoverlapping subsets
        folder_df.to_excel(vlm_eval_subset_labels_path+'.xlsx', index=False) # Correct here if you want to create multiple nonoverlapping subsets


if __name__ == '__main__':   
    
    parser = argparse.ArgumentParser(description='Create dataset subsets by sampling equally from each class')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset name to process')
    
    args = parser.parse_args()
    
    print(f"\nProcessing dataset: {args.dataset_name}")
    sample_data_subset(args.dataset_name)



