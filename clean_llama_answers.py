#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 17:19:46 2024

@author: ivan
"""

from dataset_info_and_paths import get_global_info
import os
import pandas as pd
import shutil

def clean_llama_answers():
    """
    Cleans Llama model answers by removing header tokens and end tokens.
    Makes a backup copy of original files before cleaning.
    Only processes files containing both 'llama-3' and 'answers' in filename.
    """
    # Get results folder path from global info
    results_root_folder = get_global_info()['results_root_folder_path']
    
    # Get all files in results folder
    for root, dirs, files in os.walk(results_root_folder):
        for filename in files:
            # Check if file is xlsx or csv and contains required strings
            if ('llama-3' in filename and 'answers' in filename and 
                (filename.endswith('.xlsx') or filename.endswith('.csv'))):
                
                filepath = os.path.join(root, filename)
                
                # Create backup copy with '_raw' suffix
                backup_path = filepath.rsplit('.', 1)[0] + '_raw.' + filepath.rsplit('.', 1)[1]
                shutil.copy2(filepath, backup_path)
                
                # Read file based on extension
                if filename.endswith('.xlsx'):
                    df = pd.read_excel(filepath)
                else:
                    df = pd.read_csv(filepath)
                
                # Clean text in all columns
                for col in df.columns:
                    if df[col].dtype == 'object':  # Only process text columns
                        df[col] = df[col].astype(str).apply(lambda x:
                            x.split('<|end_header_id|>')[-1] if '<|end_header_id|>' in x else x
                        ).apply(lambda x:
                            x.replace('<|eot_id|>', '') if '<|eot_id|>' in x else x
                        )
                
                # Save cleaned file
                if filename.endswith('.xlsx'):
                    df.to_excel(filepath, index=False)
                else:
                    df.to_csv(filepath, index=False)


if __name__ == '__main__':
    clean_llama_answers()