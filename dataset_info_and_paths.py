#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:32:00 2024

@author: ivan
"""

import pandas as pd
import os

def get_global_info():
    """
    Get global information about the cluster or local machine.
    
    Args:
        
    Returns:
        dict: Dictionary with global information including:
            - results_root_folder_path (str): Path to root folder for results
            - plots_root_folder_path (str): Path to root folder for plots
    """

    available_datasets = ['AML_Matek', 'Bone_Marrow_Cyto', 'WBCAtt', 'Acevedo', 'HiCervix', 'MLL23']
    available_task_types = ['0shot_classification','1shot_classification', 'nonstructured']
    available_model_families = ['gemini', 'gpt', 'llama', 'deepseek', 'medflamingo', 'llavamed']
    available_models = ['gemini-2.0-flash-exp', 'gpt-4o', 'llama-3.2-multimodal-11B', 
                        'deepseek-vl2-tiny', 'deepseek-vl2-small', 'medflamingo', 'llavamed',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-1:AwdygYO3',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-5:Awe2yBGI',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-10:AweEXbUp',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-25:AwfgA0tK',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-50:Awfmw6Cp',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-100:AwoQsgDv',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-200:AxILVLLx',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-50-auto:AwvXEh3K',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-1-auto-cell:AzMWSEqg',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-5-auto-cell:AzKy2WKx',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-10-auto-cell:AzKvO1uD',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-25-auto-cell:AzLJ9WtZ',
                        'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-50-auto-cell:Axzl8t2D']
    recommended_models = {'gemini': 'gemini-2.0-flash-exp',
                          'gpt': 'gpt-4o',
                          'llama': 'llama-3.2-multimodal-11B',
                          'deepseek': 'deepseek-vl2-small'}

    CLUSTER_ROOT_FOLDER_PATH = '/lustre/groups/labs/marr/qscd01/projects/cytology_vlm_eval'
    LOCAL_ROOT_FOLDER_PATH = '/Users/furkan.dasdelen/Library/CloudStorage/OneDrive-HelmholtzZentrumMünchen/Desktop/Marr_lab/LVLM_project'

    # Check if LOCAL_ROOT_FOLDER_PATH exists
    if os.path.exists(CLUSTER_ROOT_FOLDER_PATH):
        root_folder_path = CLUSTER_ROOT_FOLDER_PATH
        cluster_local = 'cluster'
    elif os.path.exists(LOCAL_ROOT_FOLDER_PATH):
        root_folder_path = LOCAL_ROOT_FOLDER_PATH
        cluster_local = 'local'
    else:
        raise ValueError(f"Neither {LOCAL_ROOT_FOLDER_PATH} nor {CLUSTER_ROOT_FOLDER_PATH} exists")


    vlm_eval_subsets_root_folder_path = os.path.join(root_folder_path, 'Datasets')
    github_upload_root_folder_path = os.path.join(vlm_eval_subsets_root_folder_path, 'Upload_To_Git')
    results_root_folder_path = os.path.join(root_folder_path, 'Results')
    plots_root_folder_path = os.path.join(root_folder_path, 'Plots')

    # Create results root folder if it doesn't exist
    # os.makedirs(ROOT_FOLDER_PATH, exist_ok=True)
    # os.makedirs(vlm_eval_subsets_root_folder_path, exist_ok=True)
    # os.makedirs(results_root_folder_path, exist_ok=True)
    # os.makedirs(plots_root_folder_path, exist_ok=True)

    global_info = {'cluster_local': cluster_local,
                   'root_folder_path': root_folder_path,
                   'vlm_eval_subsets_root_folder_path': vlm_eval_subsets_root_folder_path,
                   'github_upload_root_folder_path': github_upload_root_folder_path,
                   'results_root_folder_path': results_root_folder_path,
                   'plots_root_folder_path': plots_root_folder_path,
                   'available_datasets': available_datasets,
                   'available_task_types': available_task_types,
                   'available_model_families': available_model_families,
                   'available_models': available_models,
                   'recommended_models': recommended_models}
    return global_info

def get_review_model(vlm_name):
    if 'gemini' in vlm_name:
        review_model = 'gpt-4o' #vlm_name
    elif 'gpt' in vlm_name:
        review_model = 'gpt-4o' #vlm_name
    elif 'llama' in vlm_name:
        review_model = 'gpt-4o'
    elif 'deepseek' in vlm_name:
        review_model = 'gpt-4o'
    elif 'medflamingo' in vlm_name:
        review_model = 'gpt-4o'
    elif 'llavamed' in vlm_name:
        review_model = 'gpt-4o'
    else:
        raise ValueError(f"{vlm_name} not found among the models. Add to get_review_model in dataset_info_and_paths.py")
    return review_model

def get_dataset_info(dataset_name, dataset_type):
    """
    Get information about a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        dataset_type (str): train, val, test
    Returns:
        dict: Dictionary containing dataset information including:
            - dataset_name (str): Name of the dataset
            - published_dataset_location (str): URL where dataset is published
            - published_annotations_location (str): URL where annotations are published 
            - original_full_dataset_path (str): Path to the full dataset folder
            - dataset_csv_path (str): Path to CSV file with dataset info and labels
            - vlm_eval_subset_folder_path (str): Path to selected sub-dataset folder for VLM evaluation
            - vlm_eval_subset_labels_path (str): Path to the labels file for the VLM evaluation subset
            - abbreviation_dict_path (str): Path to CSV mapping label codes to names 
            - vlm_eval_subset_oline_locations (str): Path to the Oline locations of the VLM evaluation subset - if a list is given, assumes that first 1000 cells are at the first location and so on
            - results_folder_path (str): Path to save results folder            
            - plots_folder_path (str): Path to save plots folder  
            - paths_column_in_csv (str or int): Column name in the dataset CSV that contains image paths
            - n_samples_per_label (int): Number of images per class to include in subset
            - sampling_label_column_in_csv (str or int): Column name in the dataset CSV w.r.t. which the dataset will be equally drawn (for example cell types)
            - which_classes (str): Which values of sampling_label_column_in_csv to include in the dataset (e.g. 'all')
            - column_labels_to_keep (list): Which label columns from csv to include in the vlm_dataset csv
            - ground_truth_columns_conf_mat (list): Columns to use as ground truth in confusion matrix
            - predicted_columns_conf_mat (list): Columns to use as predictions in confusion matrix
            - dataset_types_to_avoid_overlap_with (list): List of dataset types (train/val/test) to avoid overlap with
            - associated_train_dataset_type (str): Type of the associated train dataset for finetuning the models
    """
    cluster_local = get_global_info()['cluster_local']

    vlm_eval_subset_folder_path = get_vlm_eval_subset_folder_path(dataset_name, dataset_type) 
    # vlm_eval_subset_folder_path = vlm_eval_subset_folder_paths['vlm_eval_subset_folder_path']
    # train_val_or_test_path = vlm_eval_subset_folder_paths['train_val_or_test_path']
    results_folder_path = get_results_folder_path(dataset_name)['results_folder_path']
    plots_folder_path = get_plots_folder_path(dataset_name)

    vlm_eval_subset_labels_path = os.path.join(vlm_eval_subset_folder_path, f'{dataset_name}_{dataset_type}_labels.csv')
    abbreviation_dict_path = os.path.join(os.path.dirname(vlm_eval_subset_folder_path), f'{dataset_name}_abbreviation_dictionary.csv')

    vlm_eval_subset_oline_locations = None

    if dataset_type == 'test':
        n_samples_per_label = 50
    elif dataset_type == 'train':
        n_samples_per_label = 10
    elif dataset_type == 'val':
        n_samples_per_label = 0

    if dataset_type == 'test':
        dataset_types_to_avoid_overlap_with = None
        associated_train_dataset_type = 'train'
    elif dataset_type == 'train':
        dataset_types_to_avoid_overlap_with = ['test']
        associated_train_dataset_type = None
    elif dataset_type == 'val':
        dataset_types_to_avoid_overlap_with = ['train', 'test']
        associated_train_dataset_type = 'train'

    

    if dataset_name == 'AML_Matek':
        published_dataset_location = 'https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/'
        published_annotations_location = published_dataset_location
        if cluster_local == 'cluster':
            original_full_dataset_path = '/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek/AML-Cytomorphology_LMU'
            dataset_csv_path = '/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek/annotations.dat'
        elif cluster_local == 'local':#
            original_full_dataset_path = None
            dataset_csv_path = '/home/ivan/Helmholtz/Furkan/Data/AML_Matek.dat'
        
        paths_column_in_csv = 0
        sampling_label_column_in_csv = 1
        which_classes = 'all' # Which classes to include in the dataset (for example in this case all cell types)
        column_labels_to_keep=[sampling_label_column_in_csv] # Which label columns from csv to include in the vlm_dataset csv
        ground_truth_columns_conf_mat = ['label']        
        predicted_columns_conf_mat = ['cell_type']
      


    elif dataset_name == 'Bone_Marrow_Cyto':
        published_dataset_location = 'https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/'
        published_annotations_location = published_dataset_location
        if cluster_local == 'cluster':
            original_full_dataset_path = '' # /lustre/groups/shared/histology_data/hematology_data/BM_cytomorphology_data/            
            dataset_csv_path = '/lustre/groups/shared/histology_data/hematology_data/BM_cytomorphology_data/bm_train.csv'
        elif cluster_local == 'local':
            original_full_dataset_path = None
            dataset_csv_path = '/home/ivan/Helmholtz/VLMevaluation/Datasets/Bone_marrow_cyto_train.csv'
        
        paths_column_in_csv = 'Image Path'
        sampling_label_column_in_csv = 'Label'
        which_classes = 'all' # Which labels to include in the dataset (for example in this case all cell types)
        column_labels_to_keep=[sampling_label_column_in_csv]
        ground_truth_columns_conf_mat = ['label']
        predicted_columns_conf_mat = ['cell_type']
 


    elif dataset_name == 'Acevedo':
        published_dataset_location = 'https://www.sciencedirect.com/science/article/pii/S2352340920303681#ec-research-data'
        published_annotations_location = published_dataset_location
        if cluster_local == 'cluster':
            original_full_dataset_path = "" # "/lustre/groups/labs/marr/qscd01/datasets/Acevedo_20"            
            dataset_csv_path = "/lustre/groups/labs/marr/qscd01/projects/cytology_vlm_eval/Datasets/Acevedo/acevedo_full_labels.csv"
        elif cluster_local == 'local':
            original_full_dataset_path = '/home/ivan/Helmholtz/Furkan/Data/Acevedo_20/'
            dataset_csv_path = '/home/ivan/Helmholtz/Furkan/Data/Acevedo_20/acevedo_full_labels.csv'


        if dataset_type == 'train':
            vlm_eval_subset_oline_locations = ['https://raw.githubusercontent.com/ivankukuljan/Acevedo_train_1/refs/heads/main/', 
                'https://raw.githubusercontent.com/ivankukuljan/Acevedo_train_2/refs/heads/main/',
                'https://raw.githubusercontent.com/ivankukuljan/Acevedo_train_3/refs/heads/main/']
            n_samples_per_label = 200
        elif dataset_type == 'val':
            vlm_eval_subset_oline_locations = 'https://raw.githubusercontent.com/ivankukuljan/Acevedo_val_1/refs/heads/main/'
            n_samples_per_label = 50
        elif dataset_type == 'test':
            vlm_eval_subset_oline_locations = ''
            n_samples_per_label = 50
            
        abbreviation_dict_path = None            

        paths_column_in_csv = 'path'
        sampling_label_column_in_csv = 'label'
        which_classes = ['Band Neutrophil',
             'Basophil',
             'Eosinophil',
             'Erythroblast',
             'Lymphocyte',
             'Metamyelocyte',
             'Monocyte',
             'Myelocyte',
             'Platelet',
             'Promyelocyte',
             'Segmented Neutrophil'] # Which labels to include in the dataset (for example in this case all cell types)
        column_labels_to_keep=[sampling_label_column_in_csv]
        ground_truth_columns_conf_mat = ['label']
        predicted_columns_conf_mat = ['cell_type']

    elif dataset_name == 'WBCAtt':
        published_dataset_location = 'https://www.sciencedirect.com/science/article/pii/S2352340920303681#ec-research-data'
        published_annotations_location = 'https://github.com/apple2373/wbcatt'
        if cluster_local == 'cluster':
            original_full_dataset_path = "/lustre/groups/labs/marr/qscd01/datasets/Acevedo_20"            
            dataset_csv_path = "/lustre/groups/labs/marr/qscd01/datasets/241211_WBCAtt/WBCAtt_morphology_annotations_train.csv"
        elif cluster_local == 'local':
            original_full_dataset_path = '/home/ivan/Helmholtz/Furkan/Data/Acevedo_20/'
            dataset_csv_path = '/home/ivan/Helmholtz/Furkan/Data/WBCAtt/WBCAtt_morphology_annotations_train.csv'
            
        abbreviation_dict_path = None

        paths_column_in_csv = 'path'
        sampling_label_column_in_csv = 'label'
        which_classes = 'all' # Which labels to include in the dataset (for example in this case all cell types)
        column_labels_to_keep=['label',
            'cell_size',
            'cell_shape', 
            'nucleus_shape',
            'nuclear_cytoplasmic_ratio',
            'chromatin_density',
            'cytoplasm_vacuole',
            'cytoplasm_texture',
            'cytoplasm_colour',
            'granule_type',
            'granule_colour',
            'granularity'
        ]
        ground_truth_columns_conf_mat = ['cell_size', 
            'cell_shape', 
            'nucleus_shape', 
            'nuclear_cytoplasmic_ratio', 
            'chromatin_density', 
            'cytoplasm_vacuole',  
            'cytoplasm_texture', 
            'cytoplasm_colour', 
            'granule_type', 
            'granule_colour', 
            'granularity']  
        # ['label', 'cell_size', 'cell_shape', 'nucleus_shape', 'nuclear_cytoplasmic_ratio', 'chromatin_density', 'cytoplasm_vacuole', 'cytoplasm_texture', 'cytoplasm_colour', 'granule_type', 'granule_colour', 'granularity']        
        predicted_columns_conf_mat = ground_truth_columns_conf_mat   
        #['label', 'cell_size', 'cell_shape', 'nucleus_shape', 'nuclear_cytoplasmic_ratio', 'chromatin_density', 'cytoplasm_vacuole', 'cytoplasm_texture', 'cytoplasm_colour', 'granule_type', 'granule_colour', 'granularity']        

    elif dataset_name == 'HiCervix':
        published_dataset_location = 'https://zenodo.org/records/11087263'
        published_annotations_location = 'https://ieeexplore.ieee.org/document/10571965'
        if cluster_local == 'cluster':
            original_full_dataset_path = ''
            dataset_csv_path = '/lustre/groups/labs/marr/qscd01/projects/cytology_vlm_eval/Datasets/HiCervix_path_labels.csv'
        elif cluster_local == 'local':
            original_full_dataset_path = None
            dataset_csv_path = '/Users/juliaschafer/Helmholtz/VLMevaluation/Datasets/papsmear_dataset_processed/HiCervix.csv'

        n_samples_per_label = 50
        abbreviation_dict_path = None

        paths_column_in_csv = 'Path'
        sampling_label_column_in_csv = 'Label'
        which_classes = 'all' 
        column_labels_to_keep=[sampling_label_column_in_csv]
        ground_truth_columns_conf_mat = ['label']
        predicted_columns_conf_mat = ['cell_type']

    elif dataset_name == 'MLL23':
        published_dataset_location = ''
        published_annotations_location = ''
        if cluster_local == 'cluster':
            original_full_dataset_path = '/lustre/groups/labs/marr/qscd01/datasets/240416_MLL23'
            dataset_csv_path = '/lustre/groups/labs/marr/qscd01/projects/cytology_vlm_eval/Datasets/MLL23_path_labels.csv'
        elif cluster_local == 'local':
            original_full_dataset_path = '/Users/furkan.dasdelen/Library/CloudStorage/OneDrive-HelmholtzZentrumMünchen/Desktop/Marr_lab/LVLM_project/Datasets/MLL23/test'
            dataset_csv_path = '/Users/furkan.dasdelen/Library/CloudStorage/OneDrive-HelmholtzZentrumMünchen/Desktop/Marr_lab/LVLM_project/Datasets/MLL23/test/MLL23_test_labels.csv'

        n_samples_per_label = 50
        abbreviation_dict_path = None

        paths_column_in_csv = 'Path'
        sampling_label_column_in_csv = 'Label'
        which_classes = 'all' 
        column_labels_to_keep=[sampling_label_column_in_csv]
        ground_truth_columns_conf_mat = ['label']
        predicted_columns_conf_mat = ['cell_type']

        
    dataset_info = {'dataset_name': dataset_name,
                    'published_dataset_location': published_dataset_location,
                    'published_annotations_location': published_annotations_location,
                    'original_full_dataset_path': original_full_dataset_path,
                    'dataset_csv_path': dataset_csv_path,
                    'vlm_eval_subset_folder_path': vlm_eval_subset_folder_path,
                    'vlm_eval_subset_labels_path': vlm_eval_subset_labels_path,
                    'abbreviation_dict_path': abbreviation_dict_path,
                    'vlm_eval_subset_oline_locations': vlm_eval_subset_oline_locations,
                    'results_folder_path': results_folder_path,
                    'plots_folder_path': plots_folder_path,  
                    'n_samples_per_label': n_samples_per_label,
                    'paths_column_in_csv': paths_column_in_csv,
                    'sampling_label_column_in_csv': sampling_label_column_in_csv,
                    'which_classes': which_classes,
                    'column_labels_to_keep': column_labels_to_keep,
                    'ground_truth_columns_conf_mat': ground_truth_columns_conf_mat,
                    'predicted_columns_conf_mat': predicted_columns_conf_mat,
                    'dataset_types_to_avoid_overlap_with': dataset_types_to_avoid_overlap_with,
                    'associated_train_dataset_type': associated_train_dataset_type}    
    return dataset_info


def get_vlm_eval_subset_folder_path(dataset_name, dataset_type): 
   """
   Get path to VLM evaluation dataset folder based on dataset and cluster/local.

   Args:
       dataset_name (str): Name of the dataset
       dataset_type (str): train, val, test
   """
   try:
       global_info = get_global_info()
       vlm_eval_subsets_root_folder_path = global_info['vlm_eval_subsets_root_folder_path']
       vlm_eval_subset_folder_path = os.path.join(vlm_eval_subsets_root_folder_path, dataset_name, dataset_type)
       if not os.path.exists(vlm_eval_subset_folder_path):
           print(f"Creating directory {vlm_eval_subset_folder_path}")
           os.makedirs(vlm_eval_subset_folder_path, exist_ok=True)
       # else:
           # print(f"Directory {vlm_eval_subset_folder_path} already exists")
       return vlm_eval_subset_folder_path
   except OSError as e:
       raise ValueError(f"Error creating directory structure: {e}")


def get_fine_tuning_subset_paths(n_train_samples_per_label, dataset_name, dataset_type, task_type, model_family):
    """
    Get path to fine tuning subsets - both jsonl and csv.
    """

    vlm_eval_subset_folder_path = get_vlm_eval_subset_folder_path(dataset_name, dataset_type)

    fine_tuning_jsonl_path = os.path.join(vlm_eval_subset_folder_path, f'fine_tuning_{dataset_name}_{dataset_type}_task_type_{task_type}_model_family_{model_family}_n_per_label_{n_train_samples_per_label}.jsonl')
    fine_tuning_csv_path = os.path.join(vlm_eval_subset_folder_path, f'fine_tuning_{dataset_name}_{dataset_type}_n_per_label_{n_train_samples_per_label}.csv')
    
    finetuning_paths = {'fine_tuning_jsonl_path': fine_tuning_jsonl_path,
                        'fine_tuning_csv_path': fine_tuning_csv_path}

    return finetuning_paths


def get_results_folder_path(dataset_name):
    """
    Get path to results folder based on dataset and cluster/local.
    
    Args:
        dataset_name (str): Name of the dataset        
    Returns:
        str: Full path to results folder
    """

    global_info = get_global_info()
    results_root_folder_path = global_info['results_root_folder_path']

    results_folder_path = os.path.join(results_root_folder_path, dataset_name)
    answers_folder_path = os.path.join(results_folder_path, 'answers')
    conf_mat_folder_path = os.path.join(results_folder_path, 'confusion_matrices')

    # Create output folder if it doesn't exist
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path, exist_ok=True)
    if not os.path.exists(answers_folder_path):
        os.makedirs(answers_folder_path, exist_ok=True)
    if not os.path.exists(conf_mat_folder_path):
        os.makedirs(conf_mat_folder_path, exist_ok=True)

    results_folders_paths = {'results_folder_path': results_folder_path,
                          'answers_folder_path': answers_folder_path,
                          'conf_mat_folder_path': conf_mat_folder_path}

    return results_folders_paths

def get_plots_folder_path(dataset_name):
    """
    Get path to plots folder based on dataset and cluster/local.
    """

    global_info = get_global_info()
    plots_root_folder_path = global_info['plots_root_folder_path']

    plots_folder_path = os.path.join(plots_root_folder_path, dataset_name)
   
    if not os.path.exists(plots_folder_path):
        os.makedirs(plots_folder_path, exist_ok=True)  

    return plots_folder_path


def get_result_path(vlm_name, dataset_name, task_type, reviewed, file_type_extension=None):
    """
    Get path to a single results file based on VLM, dataset, task type and whether it is reviewed.
    
    Args:
        vlm_name (str): Name of the vision language model
        dataset_name (str): Name of the dataset
        task_type (str): Type of task (see get_global_info()['available_task_types'])
        reviewed (bool): Whether to get path for reviewed or nonreviewed results
        file_type_extension (str, optional): File extension to append. One of 'csv', 'xlsx', or None. Defaults to None.
        
    Returns:
        dict: Dictionary containing paths to results files:
            - answers_path: Path to answers file
            - tokens_path: Path to tokens file
    """

    answers_folder_path = get_results_folder_path(dataset_name)['answers_folder_path']

    
    answers_path = os.path.join(answers_folder_path, 
                               f'{dataset_name}_{task_type}_{vlm_name}_answers')
    tokens_path = answers_path.replace('_answers', 'total_tokens_used')

    if reviewed:
        review_model = get_review_model(vlm_name)
        answers_path = answers_path + '_reviewed_' + review_model
        tokens_path = tokens_path + '_reviewed_' + review_model

    if file_type_extension != None:
        answers_path = answers_path + '.' + file_type_extension
        tokens_path = tokens_path + '.' + file_type_extension
    
    
    result_paths = {'answers_path': answers_path,
                  'tokens_path': tokens_path}

    return result_paths

def get_conf_mat_path(vlm_name, dataset_name, task_type, reviewed, file_type_extension=None):
    """
    Get path to a single confusion matrix file computed from results.

    Args:
        vlm_name (str): Name of the vision language model
        dataset_name (str): Name of the dataset
        task_type (str): Type of task (see get_global_info()['available_task_types'])
        reviewed (bool): Whether to get path for reviewed or nonreviewed results
        file_type_extension (str, optional): File extension to append. One of 'csv', 'xlsx', 'pdf', 'png', or None. Defaults to None.
        
    Returns:
        str: Full path to confusion matrix file
    """
    
    result_path = get_result_path(vlm_name, dataset_name, task_type, reviewed, file_type_extension=None)['answers_path']

    results_file_name = os.path.basename(result_path)
    results_file_name = results_file_name.replace('.csv', '').replace('.xlsx', '')
    

    conf_mat_folder_path = get_results_folder_path(dataset_name)['conf_mat_folder_path']
    plots_folder_path = get_plots_folder_path(dataset_name)
    
    conf_mat_file_name = results_file_name + '_conf_mat'
    conf_mat_plot_file_name = conf_mat_file_name + '_plot'
    conf_mat_path = os.path.join(conf_mat_folder_path, conf_mat_file_name)
    conf_mat_plot_path = os.path.join(plots_folder_path, conf_mat_plot_file_name)
    
    if file_type_extension != None:
        conf_mat_path = conf_mat_path + '.' + file_type_extension
        conf_mat_plot_path = conf_mat_plot_path + '.' + file_type_extension
    

    conf_mat_paths = {'conf_mat_path': conf_mat_path,
                      'conf_mat_plot_path': conf_mat_plot_path}
    
    return conf_mat_paths


def get_score_metrics_paths(task_type, reviewed, file_type_extension=None):
    """
    Get paths to overall score metrics files computed from results.
    
    Args:
        reviewed (bool): Whether to get path for reviewed or nonreviewed results
        file_type_extension (str, optional): File extension to append. One of 'csv', 'xlsx', or None. Defaults to None.
        
    Returns:
        dict: Dictionary containing paths to different score metric files:
            - precision_score_path: Path to precision scores file
            - sensitivity_score_path: Path to sensitivity scores file 
            - f1_score_path: Path to F1 scores file
    """

    results_root_folder_path = get_global_info()['results_root_folder_path']

    
    precision_score_path = os.path.join(results_root_folder_path, f'{task_type}_precision_scores')
    sensitivity_score_path = os.path.join(results_root_folder_path, f'{task_type}_sensitivity_scores')
    f1_score_path = os.path.join(results_root_folder_path, f'{task_type}_f1_scores')
    weighted_f1_score_path = os.path.join(results_root_folder_path, f'{task_type}_weighted_f1_scores')

    if reviewed:
        precision_score_path = precision_score_path + '_reviewed'
        sensitivity_score_path = sensitivity_score_path + '_reviewed'
        f1_score_path = f1_score_path + '_reviewed'
        weighted_f1_score_path = weighted_f1_score_path + '_reviewed'
    if file_type_extension != None:
        precision_score_path = precision_score_path + '.' + file_type_extension
        sensitivity_score_path = sensitivity_score_path + '.' + file_type_extension
        f1_score_path = f1_score_path + '.' + file_type_extension
        weighted_f1_score_path = weighted_f1_score_path + '.' + file_type_extension

    score_metrics_paths = {'precision_score_path': precision_score_path,
                          'sensitivity_score_path': sensitivity_score_path,
                          'f1_score_path': f1_score_path,
                          'weighted_f1_score_path': weighted_f1_score_path}

    return score_metrics_paths


def get_score_metrics_report_path(task_type, reviewed, file_type_extension='png'):
    """
    Get path to the overall score metrics report file computed from results.

    Args:
        reviewed (bool): Whether to get path for reviewed or nonreviewed results
        file_type_extension (str, optional): File extension to append. One of 'pdf', 'png'. Defaults to 'png'.
        
    Returns:
        str: Full path to score metrics report file
    """

    plots_root_folder_path = get_global_info()['plots_root_folder_path']
    score_metrics_report_path = os.path.join(plots_root_folder_path, f'{task_type}_score_metrics_report{"_reviewed" if reviewed else ""}.{file_type_extension}')

    return score_metrics_report_path

 
