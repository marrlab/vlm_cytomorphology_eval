#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 02:05:48 2025

@author: ivan
"""

from dataset_info_and_paths import get_global_info
import os


global_info = get_global_info()

results_root_folder_path = global_info['results_root_folder_path']

# Walk through all files in results directory
for root, dirs, files in os.walk(results_root_folder_path):
    for filename in files:
        # Check for double extensions
        if filename.endswith('.csv.csv'):
            old_path = os.path.join(root, filename)
            new_path = old_path.replace('.csv.csv', '.csv')
            print(f"Renaming {old_path} to {new_path}")
            os.rename(old_path, new_path)
            
        elif filename.endswith('.csv.xlsx'):
            old_path = os.path.join(root, filename)
            new_path = old_path.replace('.csv.xlsx', '.xlsx')
            print(f"Renaming {old_path} to {new_path}")
            os.rename(old_path, new_path)






