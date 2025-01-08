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

    CLUSTER_ROOT_FOLDER_PATH = '/lustre/groups/aih/ivan.kukuljan/VLMevaluation/' #"../"
    LOCAL_ROOT_FOLDER_PATH = '/home/ivan/Helmholtz/VLMevaluation/'

    # Check if LOCAL_ROOT_FOLDER_PATH exists
    if os.path.exists(LOCAL_ROOT_FOLDER_PATH):
        root_folder_path = LOCAL_ROOT_FOLDER_PATH
        cluster_local = 'local'
    elif os.path.exists(CLUSTER_ROOT_FOLDER_PATH):
        root_folder_path = CLUSTER_ROOT_FOLDER_PATH
        cluster_local = 'cluster'
    else:
        raise ValueError(f"Neither {LOCAL_ROOT_FOLDER_PATH} nor {CLUSTER_ROOT_FOLDER_PATH} exists")


    vlm_eval_subsets_root_folder_path = os.path.join(root_folder_path, 'Datasets')
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
                   'results_root_folder_path': results_root_folder_path,
                   'plots_root_folder_path': plots_root_folder_path}
    return global_info

def get_review_model(vlm_name):
    if 'gemini' in vlm_name:
        review_model = vlm_name
    elif 'gpt' in vlm_name:
        review_model = vlm_name
    elif 'llama' in vlm_name:
        review_model = 'gpt-4o'
    else:
        raise ValueError(f"{vlm_name} not found among the models. Add to get_review_model in dataset_info_and_paths.py")
    return review_model

def get_dataset_info(dataset_name):
    """
    Get information about a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
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
            - results_folder_path (str): Path to save results folder            
            - plots_folder_path (str): Path to save plots folder  
            - paths_column_in_csv (str or int): Column name in the dataset CSV or index that contains image paths
            - sorting_label_column_in_csv (str or int): Column name or index in the dataset CSV w.r.t. which the dataset will be equally drawn
            - which_classes (str): Which classes (for example cell types) to include in the dataset (e.g. 'all')
            - column_labels_to_keep (list): Which label columns from csv to include in the vlm_dataset csv
            - ground_truth_columns_conf_mat (list): Columns to use as ground truth in confusion matrix
            - predicted_columns_conf_mat (list): Columns to use as predictions in confusion matrix
            
    """
    cluster_local = get_global_info()['cluster_local']

    vlm_eval_subset_folder_path = get_vlm_eval_subset_folder_path(dataset_name) 
    results_folder_path = get_results_folder_path(dataset_name)
    plots_folder_path = get_plots_folder_path(dataset_name)

    vlm_eval_subset_labels_path = os.path.join(vlm_eval_subset_folder_path, f'{dataset_name}_labels.csv')

    if dataset_name == 'AML_Matek':
        published_dataset_location = 'https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/'
        published_annotations_location = published_dataset_location
        if cluster_local == 'cluster':
            original_full_dataset_path = '/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek/AML-Cytomorphology_LMU'
            dataset_csv_path = '/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek/annotations.dat'
            abbreviation_dict_path = '/lustre/groups/aih/ivan.kukuljan/VLMevaluation/Datasets/AML_Matek_abbreviation_dictionary.csv' #'/lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek/abbreviations.txt'

        elif cluster_local == 'local':#
            original_full_dataset_path = None
            dataset_csv_path = '/home/ivan/Helmholtz/Furkan/Data/AML_Matek.dat'
            abbreviation_dict_path = '/home/ivan/Helmholtz/VLMevaluation/Datasets/AML_Matek_abbreviation_dictionary.csv'

        paths_column_in_csv = 0
        sorting_label_column_in_csv = 1
        which_classes = 'all' # Which classes to include in the dataset (for example in this case all cell types)
        column_labels_to_keep=[sorting_label_column_in_csv] # Which label columns from csv to include in the vlm_dataset csv
        ground_truth_columns_conf_mat = ['1']        
        predicted_columns_conf_mat = ['cell_type']


    elif dataset_name == 'Bone_Marrow_Cyto':
        published_dataset_location = 'https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/'
        published_annotations_location = published_dataset_location
        if cluster_local == 'cluster':
            original_full_dataset_path = '' # /lustre/groups/shared/histology_data/hematology_data/BM_cytomorphology_data/            
            dataset_csv_path = '/lustre/groups/shared/histology_data/hematology_data/BM_cytomorphology_data/bm_train.csv'
            abbreviation_dict_path = '/lustre/groups/aih/ivan.kukuljan/VLMevaluation/Datasets/Bone_marrow_cyto_abbreviation_dictionary.csv'

        elif cluster_local == 'local':
            original_full_dataset_path = None
            dataset_csv_path = '/home/ivan/Helmholtz/VLMevaluation/Datasets/Bone_marrow_cyto.csv'
            abbreviation_dict_path = '/home/ivan/Helmholtz/VLMevaluation/Datasets/Bone_marrow_cyto_abbreviation_dictionary.csv'

        paths_column_in_csv = 'Image Path'
        sorting_label_column_in_csv = 'Label'
        which_classes = 'all' # Which labels to include in the dataset (for example in this case all cell types)
        column_labels_to_keep=[sorting_label_column_in_csv]
        ground_truth_columns_conf_mat = ['Label']
        predicted_columns_conf_mat = ['cell_type']

    elif dataset_name == 'WBCAtt':
        published_dataset_location = 'https://www.sciencedirect.com/science/article/pii/S2352340920303681#ec-research-data'
        published_annotations_location = 'https://github.com/apple2373/wbcatt'
        if cluster_local == 'cluster':
            original_full_dataset_path = "/lustre/groups/labs/marr/qscd01/datasets/Acevedo_20"            
            dataset_csv_path = "/lustre/groups/labs/marr/qscd01/datasets/241211_WBCAtt/WBCAtt_morphology_annotations_train.csv"
            abbreviation_dict_path = None

        elif cluster_local == 'local':
            original_full_dataset_path = '/home/ivan/Helmholtz/Furkan/Data/Acevedo_20/'
            dataset_csv_path = '/home/ivan/Helmholtz/Furkan/Data/WBCAtt/WBCAtt_morphology_annotations_train.csv'
            abbreviation_dict_path = None

        paths_column_in_csv = 'path'
        sorting_label_column_in_csv = 'label'
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

 
    
    dataset_info = {'dataset_name': dataset_name,
                    'published_dataset_location': published_dataset_location,
                    'published_annotations_location': published_annotations_location,
                    'original_full_dataset_path': original_full_dataset_path,
                    'dataset_csv_path': dataset_csv_path,
                    'vlm_eval_subset_folder_path': vlm_eval_subset_folder_path,
                    'vlm_eval_subset_labels_path': vlm_eval_subset_labels_path,
                    'abbreviation_dict_path': abbreviation_dict_path,
                    'results_folder_path': results_folder_path,
                    'plots_folder_path': plots_folder_path,  
                    'paths_column_in_csv': paths_column_in_csv,
                    'sorting_label_column_in_csv': sorting_label_column_in_csv,
                    'which_classes': which_classes,
                    'column_labels_to_keep': column_labels_to_keep,
                    'ground_truth_columns_conf_mat': ground_truth_columns_conf_mat,
                    'predicted_columns_conf_mat': predicted_columns_conf_mat}    
    return dataset_info


def get_vlm_eval_subset_folder_path(dataset_name): #, structured_nonstructured
    """
    Get path to VLM evaluation dataset folder based on dataset and cluster/local.
    """

    global_info = get_global_info()
    vlm_eval_subsets_root_folder_path = global_info['vlm_eval_subsets_root_folder_path']

    vlm_eval_subset_folder_path = os.path.join(vlm_eval_subsets_root_folder_path, dataset_name) #+'_'+structured_nonstructured)

    os.makedirs(vlm_eval_subset_folder_path, exist_ok=True)

    return vlm_eval_subset_folder_path


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

    # Create output folder if it doesn't exist
    os.makedirs(results_folder_path, exist_ok=True)

    return results_folder_path

def get_plots_folder_path(dataset_name):
    """
    Get path to plots folder based on dataset and cluster/local.
    """

    global_info = get_global_info()
    plots_root_folder_path = global_info['plots_root_folder_path']

    plots_folder_path = os.path.join(plots_root_folder_path, dataset_name)

    os.makedirs(plots_folder_path, exist_ok=True)

    return plots_folder_path


def get_result_path(vlm_name, dataset_name, structured_nonstructured_review, answers_tokens, csv_xlsc_none=None):
    """
    Get path to results file based on VLM, dataset and review type.
    
    Args:
        vlm_name (str): Name of the vision language model
        dataset_name (str): Name of the dataset
        structured_nonstructured_review (str): Type of review ('structured', 'nonstructured', 'review')
        answers_tokens (str): Whether to get path for 'answers' or 'tokens' file
        csv_xlsc_none (str, optional): File extension to append. One of 'csv', 'xlsx', or None. Defaults to None.
        
    Returns:
        str: Full path to results CSV file
    """

    results_folder_path = get_results_folder_path(dataset_name)

    if answers_tokens == 'tokens':
        answers_tokens = 'total_tokens_used'

    elif answers_tokens == 'answer':
        answers_tokens = 'answers'

    if structured_nonstructured_review in ['structured', 'review', 'reviewed']:
        structured_nonstructured = 'structured'
    elif structured_nonstructured_review == 'nonstructured':
        structured_nonstructured = 'nonstructured'

    result_path = os.path.join(results_folder_path, 
                               f'{dataset_name}_{structured_nonstructured}_{vlm_name}_{answers_tokens}')

    if structured_nonstructured_review in ['review', 'reviewed']:
        review_model = get_review_model(vlm_name)
        result_path = result_path + '_reviewed_' + review_model

    if csv_xlsc_none == 'csv':
        result_path = result_path + '.csv'
    elif csv_xlsc_none == 'xlsx':
        result_path = result_path + '.xlsx'
    elif csv_xlsc_none == 'none':
        result_path = result_path

    return result_path

def get_conf_mat_path(vlm_name, dataset_name, reviewed_yes_no, csv_xlsc_none=None):
    """
    Get path to confusion matrix file computed from results.
    """
    

    if reviewed_yes_no == 'yes':
        structured_nonstructured_review = 'review'
    elif reviewed_yes_no == 'no':
        structured_nonstructured_review = 'structured'


    result_path = get_result_path(vlm_name, dataset_name, structured_nonstructured_review, 'answers', csv_xlsc_none)
    
    result_path_without_extension = result_path.replace('.csv', '').replace('.xlsx', '')
    
    conf_mat_path = result_path_without_extension + '_conf_mat'
    
    if csv_xlsc_none == 'csv':
        conf_mat_path = conf_mat_path + '.csv'
    elif csv_xlsc_none == 'xlsx':
        conf_mat_path = conf_mat_path + '.xlsx'
    elif csv_xlsc_none == 'none':
        conf_mat_path = conf_mat_path

    return conf_mat_path

def get_score_metrics_paths(reviewed_yes_no, csv_xlsc_none=None):
    """
    Get paths to score metrics files computed from results.
    
    Args:
        reviewed_yes_no (str): Whether to get path for 'yes' or 'no' reviewed results
        csv_xlsc_none (str, optional): File extension to append. One of 'csv', 'xlsx', or None. Defaults to None.
        
    Returns:
        dict: Dictionary containing paths to different score metric files:
            - precision_score_path: Path to precision scores file
            - sensitivity_score_path: Path to sensitivity scores file 
            - f1_score_path: Path to F1 scores file
    """

    results_root_folder_path = get_global_info()['results_root_folder_path']

    
    precision_score_path = os.path.join(results_root_folder_path, 'precision_scores')
    sensitivity_score_path = os.path.join(results_root_folder_path, 'sensitivity_scores')
    f1_score_path = os.path.join(results_root_folder_path, 'f1_scores')

    if reviewed_yes_no == 'yes':
        precision_score_path = precision_score_path + '_reviewed'
        sensitivity_score_path = sensitivity_score_path + '_reviewed'
        f1_score_path = f1_score_path + '_reviewed'

    score_metrics_paths = {'precision_score_path': precision_score_path,
                          'sensitivity_score_path': sensitivity_score_path,
                          'f1_score_path': f1_score_path
                          }

    return score_metrics_paths

def get_conf_mat_plot_path(vlm_name, dataset_name, reviewed_yes_no, save_format_pdf_png='png'):
    """
    Get path to confusion matrix plot file computed from results.
    """

    conf_mat_path = get_conf_mat_path(vlm_name, dataset_name, reviewed_yes_no, csv_xlsc_none=None)

    plots_folder_path = get_plots_folder_path(dataset_name)
    
    conf_mat_plot_path = os.path.join(plots_folder_path, os.path.basename(conf_mat_path).replace('.csv', '').replace('.xlsx', '') + '_plot.' + save_format_pdf_png)

    return conf_mat_plot_path

def get_score_metrics_report_path(reviewed_yes_no, save_format_pdf_png='png'):
    """
    Get path to score metrics report file computed from results.
    """

    plots_root_folder_path = get_global_info()['plots_root_folder_path']
    score_metrics_report_path = os.path.join(plots_root_folder_path, f'score_metrics_report{"_reviewed" if reviewed_yes_no == "yes" else ""}.{save_format_pdf_png}')

    return score_metrics_report_path

 