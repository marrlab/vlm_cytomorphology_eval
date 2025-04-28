import pandas as pd
import numpy as np
import os
import shutil       
from dataset_info_and_paths import get_dataset_info, get_result_path, get_vlm_eval_subset_folder_path



def select_images_for_Michele_review(number_per_type_and_correctness, vlm_name, dataset_name, task_type='nonstructured'):
    dataset_info = get_dataset_info(dataset_name, 'test')

    groud_truth_labels_path = dataset_info['vlm_eval_subset_labels_path']

    predicted_labels_path = get_result_path(vlm_name, dataset_name, task_type, reviewed=True, file_type_extension='csv')['answers_path']
    predicted_labels_path = predicted_labels_path.replace(':', '_')
    
    answers_path = get_result_path(vlm_name, dataset_name, task_type, reviewed=False, file_type_extension='csv')['answers_path']
    
    testset_folder_path = get_vlm_eval_subset_folder_path(dataset_name, 'test')
    
    Michele_data_folder_path = get_vlm_eval_subset_folder_path(dataset_name, 'Michele_'+vlm_name)
    
    # Michele_data_folder_path = Michele_data_folder_path.replace(':', '_')
    
    os.makedirs(Michele_data_folder_path, exist_ok=True)

    ground_truth_df = pd.read_csv(groud_truth_labels_path)
    predicted_labels_df = pd.read_csv(predicted_labels_path)
    answers_df = pd.read_csv(answers_path)

    all_cell_types = ground_truth_df['label'].unique()

    df_for_Michele = pd.DataFrame(columns=['image_name', 'correct_cell_type', 'correctness (1-excellent, 2-good, 3-ok, 4-poor, 5-completely wrong)', 'mistakes', 'LLM_description'])
    df_for_us = pd.DataFrame(columns=['image_name', 'original_image_name', 'ground_truth_cell_type', 'predicted_cell_type', 'Michele_cell_type', 'correctness (1-excellent, 2-good, 3-ok, 4-poor, 5-completely wrong)', 'mistakes', 'LLM_description'])

    cell_count=0

    def enter_row(cell_type,correct_wrong):
        nonlocal cell_count
        nonlocal df_for_Michele
        nonlocal df_for_us
        nonlocal ground_truth_df

        ground_truth_cell_type_df = ground_truth_df[ground_truth_df['label'] == cell_type]
        if len(ground_truth_cell_type_df) == 0:
            print(f'No ground truth images found for cell type {cell_type}. Skipping...')
            return

        predicted_cell_type_df = predicted_labels_df[predicted_labels_df['image_name'].isin(ground_truth_cell_type_df['image_name'])]
        if correct_wrong == 'correct':
            predicted_cell_type_df = predicted_cell_type_df[predicted_cell_type_df['cell_type'] == cell_type]
        elif correct_wrong == 'wrong':
            predicted_cell_type_df = predicted_cell_type_df[predicted_cell_type_df['cell_type'] != cell_type]
        else:
            raise ValueError('correct_wrong must be either "correct" or "wrong"')

        if len(predicted_cell_type_df) == 0:
            print(f'No predicted {correct_wrong} images found for cell type {cell_type}. Skipping...')
            return

        # Select random image from predicted cell type
        original_image_name = predicted_cell_type_df.sample(n=1)['image_name'].values[0]
        
        ground_truth_label = ground_truth_cell_type_df[ground_truth_cell_type_df['image_name'] == original_image_name]['label'].values[0]
        predicted_label = predicted_cell_type_df[predicted_cell_type_df['image_name'] == original_image_name]['cell_type'].values[0]
        LLM_description = answers_df[answers_df['image_name'] == original_image_name]['cell_type'].values[0]

        Michele_image_name = 'cell_'+str(cell_count)
        
        cell_count += 1

        Michele_image_path = os.path.join(Michele_data_folder_path, Michele_image_name+'.png')

        df_for_Michele_row = pd.DataFrame({'image_name': Michele_image_name, 'correct_cell_type': '', 'correctness (1-excellent, 2-good, 3-ok, 4-poor, 5-completely wrong)': np.nan, 'mistakes': '', 'LLM_description': LLM_description}, index=[0])

        df_for_us_row = pd.DataFrame({'image_name': Michele_image_name, 'original_image_name': original_image_name, 'ground_truth_cell_type': ground_truth_label, 'predicted_cell_type': predicted_label, 'Michele_cell_type': '', 'correctness (1-excellent, 2-good, 3-ok, 4-poor, 5-completely wrong)': np.nan, 'mistakes': '', 'LLM_description': LLM_description}, index=[0])


        df_for_Michele = pd.concat([df_for_Michele, df_for_Michele_row], ignore_index=True)
        df_for_us = pd.concat([df_for_us, df_for_us_row], ignore_index=True)

        # Copy image from test folder to Michele folder with new name
        try:
            original_image_path = os.path.join(testset_folder_path, original_image_name+'.jpg')
            Michele_image_path = os.path.join(Michele_data_folder_path, Michele_image_name+'.jpg')
            shutil.copy2(original_image_path, Michele_image_path)
        except:
            try:
                original_image_path = os.path.join(testset_folder_path, original_image_name+'.png')
                Michele_image_path = os.path.join(Michele_data_folder_path, Michele_image_name+'.png')
                shutil.copy2(original_image_path, Michele_image_path)
            except:
                print(f'Image {original_image_name} not found in {testset_folder_path}')


        # Remove the row with original_image_name from ground_truth_df
        ground_truth_df = ground_truth_df[ground_truth_df['image_name'] != original_image_name]

        return



    for i in range(number_per_type_and_correctness):

        # Incorrectly classified images
        for cell_type in all_cell_types:
            enter_row(cell_type, 'wrong')

        # Randomly permute all_cell_types
        np.random.shuffle(all_cell_types)

        # Correctly classified images
        for cell_type in all_cell_types:
            enter_row(cell_type, 'correct')

        # Randomly permute all_cell_types
        np.random.shuffle(all_cell_types)


    # Save the dataframes to Excel
    df_for_Michele.to_excel(os.path.join(Michele_data_folder_path, 'Michele_review.xlsx'), index=False)
    df_for_us.to_excel(os.path.join(Michele_data_folder_path, 'us_complete_data.xlsx'), index=False)

    return


number_per_type_and_correctness = 5

vlm_name = 'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-200:AxILVLLx'

dataset_name = 'Acevedo'


select_images_for_Michele_review(number_per_type_and_correctness, vlm_name, dataset_name, task_type='nonstructured')


