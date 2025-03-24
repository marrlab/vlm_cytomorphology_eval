#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 13:19:32 2025

@author: ivan
"""

import pandas as pd
import numpy as np
from dataset_info_and_paths import get_result_path, get_explainability_plot_path, get_explainability_plot_vertical_path
import matplotlib.pyplot as plt

def plot_explainability_averaged_scores(dataset_name: str, vlm_names: list[str], vlm_labels: list[str], cell_type = None, ground_truth_or_predicted = 'predicted', vlm_name_to_sort = None, no_features_to_plot = None, reviewed=True, figsize = (10, 6)):
    """
    Plot the averaged scores for all VLM models for the explainability task.
    """

    linestyles = ('-', '--', '-.', ':')
    markers = ('o', 's', 'D', 'P')
    markerfacecolors = ('k', 'none', 'k', 'none')

    results = ()
    for vlm_name in vlm_names:
        vlm_name = vlm_name.replace(':', '_')
        results_path = get_result_path(vlm_name, dataset_name, task_type = 'explainability', reviewed=reviewed, file_type_extension='csv')['answers_path']
        results = results + (pd.read_csv(results_path),)

    # Get all column names except image_name, ground_truth_label, predicted_label
    features = np.array([col for col in results[0].columns if col not in ['image_name', 'ground_truth_label', 'predicted_label']])

    if no_features_to_plot != None:
        features = features[:no_features_to_plot]

    if cell_type != None:
        if (ground_truth_or_predicted == 'predicted') or (ground_truth_or_predicted == 'predicted_label'):
            sort_column = 'predicted_label'
        elif (ground_truth_or_predicted == 'ground_truth') or (ground_truth_or_predicted == 'ground_truth_label'):
            sort_column = 'ground_truth_label'
        else:
            raise ValueError(f"Invalid value for ground_truth_or_predicted: {ground_truth_or_predicted}")
        
        ground_truth_or_predicted = sort_column

        results_nonfiltered = results

        results = ()

        for i in range(len(vlm_names)): 
            results = results + (results_nonfiltered[i][results_nonfiltered[i][sort_column] == cell_type],)  
    else:
        cell_type = 'all'
  
    avg_scores = ()
    std_scores = ()

    for i in range(len(vlm_names)):    

        avg_scores = avg_scores + (results[i][features].mean(axis=0).values,)
        std_scores = std_scores + (results[i][features].std(axis=0).values,)

    sort_model_ind = 0

    if vlm_name_to_sort != None:
        sort_model_ind = vlm_names.index(vlm_name_to_sort)

    # Find indices that would sort avg_scores for the specified model
    sort_indices = (-avg_scores[sort_model_ind]).argsort()

    # Sort all avg_scores and std_scores arrays using these indices
    avg_scores = tuple(scores[sort_indices] for scores in avg_scores)
    std_scores = tuple(scores[sort_indices] for scores in std_scores)
    features = features[sort_indices]

    explainability_plot_path = get_explainability_plot_path(dataset_name, cell_type, ground_truth_or_predicted, reviewed, file_type_extension='png')
    
    # Extract feature names before first ' ('
    features = np.array([feature.split(' (')[0] for feature in features])
    # Replace 'Segmentation' with 'Segm.' and 'Nuclear' with 'Nucl.' in feature names
    features = np.array([feature.replace('Segmentation', 'Segm.').replace('Nuclear', 'Nucl.') for feature in features])

    plt.figure(figsize=figsize)
    
    for i, vlm_name in enumerate(vlm_names):
        # print(avg_scores[i])
        # print(std_scores[i])
        plt.errorbar(range(len(avg_scores[i])), avg_scores[i], yerr=std_scores[i], fmt=markers[i], markerfacecolor=markerfacecolors[i], capsize=10, markersize=12, label=vlm_labels[i], color='k')

    # plt.xlabel('Feature Importance')
    plt.ylabel('Averaged Score')
    if cell_type != 'all':
        plt.title(f'{cell_type}')
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    if cell_type == 'all':
        plt.legend()

    plt.tight_layout()
    plt.savefig(explainability_plot_path, dpi=300)
    plt.show()


    

   
# dataset_name = 'Acevedo'
# reviewed = True

# vlm_name = 'gpt-4o'
# fine_tuned_vlm_name = 'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-200:AxILVLLx'
# vlm_names = [vlm_name, fine_tuned_vlm_name]
# vlm_labels = ['GPT-4o', 'finetuned GPT-4o, n=200']
# vlm_name_to_sort = fine_tuned_vlm_name

# cell_types = []
# for vlm_name in vlm_names:
#     vlm_name = vlm_name.replace(':', '_')
#     results_path = get_result_path(vlm_name, dataset_name, task_type = 'explainability', reviewed=reviewed, file_type_extension='csv')['answers_path']
#     results = pd.read_csv(results_path)
#     cell_types_results = np.unique(results['ground_truth_label']).tolist()
#     cell_types = list(set(cell_types).union(set(cell_types_results)))


# figsize = (10, 6)
# cell_type = None

# plot_explainability_averaged_scores(dataset_name, vlm_names, vlm_labels, cell_type = cell_type, vlm_name_to_sort = vlm_name_to_sort, reviewed=reviewed, figsize = figsize)


# ground_truth_or_predicted = 'predicted_label'

# cell_types = []
# for vlm_name in vlm_names:
#     vlm_name = vlm_name.replace(':', '_')
#     results_path = get_result_path(vlm_name, dataset_name, task_type = 'explainability', reviewed=reviewed, file_type_extension='csv')['answers_path']
#     results = pd.read_csv(results_path)
#     cell_types_results = results[ground_truth_or_predicted].astype(str).unique().tolist()
#     cell_types = list(set(cell_types).union(set(cell_types_results)))


# no_features_to_plot = 4
# figsize = (3, 3)

# for cell_type in cell_types:
#     plot_explainability_averaged_scores(dataset_name, vlm_names, vlm_labels, cell_type = cell_type, ground_truth_or_predicted = ground_truth_or_predicted,vlm_name_to_sort = vlm_name_to_sort, no_features_to_plot = no_features_to_plot,reviewed=reviewed, figsize = figsize)



def plot_explainability_averaged_scores_vertical(dataset_name: str, vlm_names: list[str], vlm_labels: list[str], cell_types: list[str], ground_truth_or_predicted = 'predicted', vlm_name_to_sort = None, reviewed=True, figsize = (10, 10)):
    """
    Plot the averaged scores for all VLM models for the explainability task.
    """

    linestyles = ('-', '--', '-.', ':')
    markers = ('o', 's', 'D', 'P')
    markerfacecolors = ('k', 'none', 'k', 'none')

    results = ()
    for vlm_name in vlm_names:
        vlm_name = vlm_name.replace(':', '_')
        results_path = get_result_path(vlm_name, dataset_name, task_type = 'explainability', reviewed=reviewed, file_type_extension='csv')['answers_path']
        results = results + (pd.read_csv(results_path),)

    # Get all column names except image_name, ground_truth_label, predicted_label
    features = np.array([col for col in results[0].columns if col not in ['image_name', 'ground_truth_label', 'predicted_label']])

    if (ground_truth_or_predicted == 'predicted') or (ground_truth_or_predicted == 'predicted_label'):
        sort_column = 'predicted_label'
    elif (ground_truth_or_predicted == 'ground_truth') or (ground_truth_or_predicted == 'ground_truth_label'):
        sort_column = 'ground_truth_label'
    else:
        raise ValueError(f"Invalid value for ground_truth_or_predicted: {ground_truth_or_predicted}")

    sort_model_ind = 0

    if vlm_name_to_sort != None:
        sort_model_ind = vlm_names.index(vlm_name_to_sort)
    
    ground_truth_or_predicted = sort_column

    # Extract feature names before first ' ('
    features_labels = np.array([feature.split(' (')[0] for feature in features])
    # Replace 'Segmentation' with 'Segm.' and 'Nuclear' with 'Nucl.' in feature names
    #features_labels = np.array([feature.replace('Segmentation', 'Segm.').replace('Nuclear', 'Nucl.') for feature in features_labels])

    explainability_plot_vertical_path = get_explainability_plot_vertical_path(dataset_name, ground_truth_or_predicted, reviewed, file_type_extension='png')

    # if no_features_to_plot != None:
    #    features = features[:no_features_to_plot]

    n_cell_types = len(cell_types)
    fig, axs = plt.subplots(1,n_cell_types, figsize=figsize)
    if n_cell_types == 1:
        axs = [axs]

    for i, cell_type in enumerate(cell_types):

        if (cell_type != None) and (cell_type != 'all'):   

            results_filtered = ()

            for j in range(len(vlm_names)): 
                results_filtered = results_filtered + (results[j][results[j][sort_column] == cell_type],)  
        else:
            results_filtered = results
            cell_type = 'All cell types'
    
        avg_scores = ()
        std_scores = ()

        for j in range(len(vlm_names)):    
            avg_scores = avg_scores + (results_filtered[j][features].mean(axis=0).values,)
            std_scores = std_scores + (results_filtered[j][features].std(axis=0).values,)     

        if i == 0:
            # Find indices that would sort avg_scores for the specified model
            sort_indices = (-avg_scores[sort_model_ind]).argsort()
            features = features[sort_indices]
            features_labels = features_labels[sort_indices]

        # Sort all avg_scores and std_scores arrays using these indices
        avg_scores = tuple(scores[sort_indices] for scores in avg_scores)
        std_scores = tuple(scores[sort_indices] for scores in std_scores)
        
        for j, vlm_name in enumerate(vlm_names):
            # print(avg_scores[i])
            # print(std_scores[i])
            axs[i].errorbar(avg_scores[j], range(len(avg_scores[j])), xerr=std_scores[j], fmt=markers[j], markerfacecolor=markerfacecolors[j], capsize=10, markersize=12, label=vlm_labels[j], color='k')

        axs[i].invert_yaxis()

        # plt.xlabel('Feature Importance')
        axs[i].set_xlabel('Averaged Score')
        #if cell_type != 'all':        
        axs[i].set_title(f'{cell_type}', fontsize=14)
        if i==0:
            axs[i].set_yticks(range(len(features_labels)), features_labels, rotation=0, ha='right', fontsize=14)
        else:
            axs[i].set_yticks([])
            
        # if cell_type == 'All cell types':
        if i == len(cell_types) - 1:
            axs[i].legend()

    plt.tight_layout()
    plt.savefig(explainability_plot_vertical_path, dpi=300)
    plt.show()
    
    


dataset_name = 'Acevedo'
reviewed = True
figsize = (18, 10)
vlm_name = 'gpt-4o'
fine_tuned_vlm_name = 'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-200:AxILVLLx'
vlm_names = [vlm_name, fine_tuned_vlm_name]
vlm_labels = ['GPT-4o', 'finetuned GPT-4o, n=200']
vlm_name_to_sort = fine_tuned_vlm_name   
# ground_truth_or_predicted = 'ground_truth_label' #'predicted_label'

cell_types = ['all', 'Segmented Neutrophil', 'Eosinophil', 'Basophil', 'Platelet']

for ground_truth_or_predicted in ['ground_truth_label', 'predicted_label']:
    plot_explainability_averaged_scores_vertical(dataset_name, vlm_names, vlm_labels, cell_types, ground_truth_or_predicted = ground_truth_or_predicted, vlm_name_to_sort = vlm_name_to_sort, reviewed=reviewed, figsize = figsize)

