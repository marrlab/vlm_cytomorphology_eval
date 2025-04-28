import pandas as pd
import numpy as np
import os
from dataset_info_and_paths import get_global_info, get_dataset_info, get_result_path, get_score_metrics_paths
from plot_conf_mat import compute_confusion_matrix, compute_score_metrics




def split_answers_into_folds(vlm_name, dataset_name, task_type, reviewed, how_many_folds):
    """
    Split the answers into folds and save them in the results folder.
    """
    
    dataset_info = get_dataset_info(dataset_name, 'test')
    
    ground_truth_df_path = dataset_info['vlm_eval_subset_labels_path']
    ground_truth_columns = dataset_info['ground_truth_columns_conf_mat']

    full_answers_path = get_result_path(vlm_name, dataset_name, task_type, reviewed, file_type_extension='csv', fold_number=None)['answers_path']

    
    full_answers_df = pd.read_csv(full_answers_path)
    ground_truth_df = pd.read_csv(ground_truth_df_path)

    # Remove .jpg and .png extensions from image names
    full_answers_df['image_name'] = full_answers_df['image_name'].str.replace('.jpg', '').str.replace('.png', '')

    cell_types = pd.unique(ground_truth_df[ground_truth_columns[0]].values.flatten())

    # Split the answers into folds

    folds = []

    for i in range(how_many_folds):
        folds.append(pd.DataFrame(columns=full_answers_df.columns))

    for cell_type in cell_types:
        images_for_cell_type = ground_truth_df[(ground_truth_df[ground_truth_columns[0]] == cell_type).values.flatten()]['image_name'].tolist()
        
    
        # Calculate size of each fold for this cell type
        fold_size = len(images_for_cell_type) // how_many_folds
        remainder = len(images_for_cell_type) % how_many_folds

        # Split images into folds
        start = 0
        for i in range(how_many_folds):
            # Add one extra image to early folds if there's a remainder
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end = start + current_fold_size
            
            
            # Get images for this fold
            fold_images = images_for_cell_type[start:end]
            
            # Add rows for these images to the fold's dataframe
            fold_rows = full_answers_df[full_answers_df['image_name'].isin(fold_images)]
            folds[i] = pd.concat([folds[i], fold_rows])
            
            start = end

    for i in range(how_many_folds):
        folds[i].to_csv(get_result_path(vlm_name, dataset_name, task_type, reviewed, file_type_extension='csv', fold_number=i)['answers_path'], index=None)
        folds[i].to_csv(get_result_path(vlm_name, dataset_name, task_type, reviewed, file_type_extension='xlsx', fold_number=i)['answers_path'], index=None)

    return folds

def compute_mean_and_std_of_folds_score_metrics(task_type, reviewed):
    """
    Compute the mean and standard deviation of the score metrics of the folds.
    """
    
    score_metrics_paths = get_score_metrics_paths(task_type, reviewed, fold_number=0, file_type_extension=None)

    for key in score_metrics_paths.keys():
        path = score_metrics_paths[key]        

        if os.path.exists(path + '.csv'):
            scores_df = pd.read_csv(path + '.csv', index_col=0)
        elif os.path.exists(path + '.xlsx'):
            scores_df = pd.read_excel(path + '.xlsx', index_col=0)

        vlm_names = scores_df.columns.tolist()
        dataset_names = scores_df.index.str.split(',').str[0].unique().tolist()


        means_df = pd.DataFrame(columns=vlm_names)
        
        stds_df = pd.DataFrame(columns=vlm_names)

        max_folds = 0

        for dataset_name in dataset_names:
            scores_dataset_df = scores_df[scores_df.index.str.contains(dataset_name)]
            if len(scores_dataset_df) == 0:
                print(f"No scores found for {dataset_name} for {key}")
                continue

            max_folds = max(max_folds, len(scores_dataset_df))

            means_dataset_df = pd.DataFrame(scores_dataset_df.mean(axis=0)).T
            means_dataset_df.index = [dataset_name]
            stds_dataset_df = pd.DataFrame(scores_dataset_df.std(axis=0)).T
            stds_dataset_df.index = [dataset_name]

            means_df = pd.concat([means_df, means_dataset_df])
            stds_df = pd.concat([stds_df, stds_dataset_df])

        means_path = path + f'_means_{max_folds}_folds'
        stds_path = path + f'_stds_{max_folds}_folds'

        means_df.to_csv(means_path + '.csv', index=True)
        means_df.to_excel(means_path + '.xlsx', index=True)
        stds_df.to_csv(stds_path + '.csv', index=True)
        stds_df.to_excel(stds_path + '.xlsx', index=True)
  



if __name__ == '__main__':
    how_many_folds=5
    
    compute_confmat_from_scratch = True
    global_info = get_global_info()
    for task_type in [t for t in global_info['available_task_types'] if t != 'nonstructured']:
        for dataset_name in global_info['available_datasets']:
            # for vlm_family in global_info['available_model_families']:
            #     vlm_name = global_info['recommended_models'][vlm_family]
            for vlm_name in ['biomedclip']: # global_info['available_models']:            
                for reviewed in [True, False]:
                    try:
                        split_answers_into_folds(vlm_name, dataset_name, task_type, reviewed, how_many_folds)
                        for i in range(how_many_folds):
                            compute_score_metrics(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch=True, fold_number=i, save_overall_metrics=True)
                    except Exception as e:
                        print(f"Error computing fold confusion matrix for {vlm_name} on {dataset_name} for {task_type} with {reviewed} reviews: {e}")
        for reviewed in [True, False]:
            try:
                compute_mean_and_std_of_folds_score_metrics(task_type, reviewed)
            except Exception as e:
                print(f"Error computing mean and std of folds score metrics for {task_type} with {reviewed} reviews: {e}")

 

# vlm_name = 'deepseek-vl2-small'
# dataset_name = 'HiCervix'
# task_type = '0shot_classification'
# reviewed = True
# how_many_folds = 5

# split_answers_into_folds(vlm_name, dataset_name, task_type, reviewed, how_many_folds)

# for i in range(how_many_folds):
#     compute_score_metrics(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch=True, fold_number=i, save_overall_metrics=True)
    
    

# vlm_name = 'llavamed'
# dataset_name = 'Bone_Marrow_Cyto'
# task_type = '0shot_classification'
# reviewed = True
# how_many_folds = 5

# split_answers_into_folds(vlm_name, dataset_name, task_type, reviewed, how_many_folds)

# for i in range(how_many_folds):
#     compute_score_metrics(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch=True, fold_number=i, save_overall_metrics=True)