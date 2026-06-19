import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from vlm_cytomorphology_eval.config.dataset_info_and_paths import get_global_info, get_dataset_info, get_result_path, get_conf_mat_path, get_score_metrics_paths, get_score_metrics_report_path

def compute_confusion_matrix(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch=True, fold_number=None):
    """
    Compute confusion matrix comparing ground truth and predicted labels.
    
    Args:
        vlm_name (str): Name of the vision language model
        dataset_name (str): Name of the dataset
        task_type (str): Type of task (see get_global_info()['available_task_types'])
        reviewed (bool): Whether to compute based on reviewed results or not
        compute_confmat_from_scratch (bool): Compute the confusion matrix from scratch or not (default: False)
    Returns:
        pd.DataFrame: Confusion matrix as a dataframe
    """
    
    output_path = get_conf_mat_path(vlm_name, dataset_name, task_type, reviewed, file_type_extension=None)['conf_mat_path']
    
    if (os.path.exists(output_path+'.csv')) and (not compute_confmat_from_scratch):
        print(f"Confusion matrix already exists at {output_path}. Skipping computation.")
        conf_matrix = pd.read_csv(output_path+'.csv')
        return conf_matrix
    elif (os.path.exists(output_path+'.xlsx')) and (not compute_confmat_from_scratch):
        print(f"Confusion matrix already exists at {output_path}. Skipping computation.")
        conf_matrix = pd.read_excel(output_path+'.xlsx')
        return conf_matrix
    else:
        dataset_info = get_dataset_info(dataset_name, 'test')
    
        ground_truth_df_path = dataset_info['vlm_eval_subset_labels_path']
        ground_truth_columns = dataset_info['ground_truth_columns_conf_mat']
        predicted_df_path = get_result_path(vlm_name, dataset_name, task_type, reviewed, file_type_extension='csv', fold_number=fold_number)['answers_path']
        predicted_columns = dataset_info['predicted_columns_conf_mat']
        labels_dict_path = dataset_info['abbreviation_dict_path']
        # print(' ')
        # print('----------------------------------------')
        # print(predicted_df_path)
        # print('----------------------------------------')
        # print(' ')
        
        # Check if ground truth and predicted columns have same length
        if len(ground_truth_columns) != len(predicted_columns):
            raise ValueError(f"Number of ground truth columns ({len(ground_truth_columns)}) does not match number of predicted columns ({len(predicted_columns)})")

        
        # print(f"predicted_df_path path: {predicted_df_path}")
        
        # print(ground_truth_df_path)
    
        # Load the dataframes
        ground_truth_df = pd.read_csv(ground_truth_df_path)
        
        predicted_df = pd.read_csv(predicted_df_path)
        
        # print('-------> Found!')
        # print(predicted_df)

        # Rename ground truth columns to match predicted columns
        rename_dict = dict(zip(ground_truth_columns, predicted_columns))
        ground_truth_df = ground_truth_df.rename(columns=rename_dict)

        ground_truth_columns = predicted_columns
    
        # Load labels dictionary 
        if labels_dict_path:
            labels_dict = pd.read_csv(labels_dict_path, sep=',', header=None, names=['code', 'name'])
        else:
            labels_dict = None
        
        # Get unique labels for each column
        ground_truth_label_sets = {col: set(ground_truth_df[col].unique()) for col in ground_truth_columns}
        
        # Create multi-index for rows and columns
        row_tuples = [(col, label) for col in ground_truth_columns 
                      for label in sorted(ground_truth_label_sets[col])]
        col_tuples = row_tuples.copy()
        # col_tuples = [(col, label) for col in predicted_columns 
        #               for label in sorted(set().union(*ground_truth_label_sets.values()))]
        col_tuples.append(('all', 'not id'))  # Single 'not id' column for all predictions
        
        row_index = pd.MultiIndex.from_tuples(row_tuples, names=['column', 'label'])
        col_index = pd.MultiIndex.from_tuples(col_tuples, names=['column', 'label'])
        
        # Initialize confusion matrix with zeros
        conf_matrix = pd.DataFrame(0, index=row_index, columns=col_index)
        
        # Fill confusion matrix
        for _, row in ground_truth_df.iterrows():
            for i, gt_col in enumerate(ground_truth_columns):
                gt_label = row[gt_col]
                # Find corresponding prediction
                pred = predicted_df[predicted_df['image_name'] == row['image_name']]
                
                if len(pred) == 0:
                    pred = predicted_df[predicted_df['image_name'] == row['image_name']+'.jpg']
                    
                if len(pred) == 0:
                    pred = predicted_df[predicted_df['image_name'] == row['image_name']+'.png']
                
                
                # print(row['image_name'])
                # print(predicted_df['image_name'])
                
                if len(pred) == 0:
                    continue
                    
                # Check if any code appears directly in the answer
                pred_answer = pred.iloc[0][predicted_columns[i]]
                
                pred_label = None
                
                if labels_dict is not None:
                    # First check for direct code matches
                    for _, label_row in labels_dict.iterrows():
                        if isinstance(pred_answer, str):
                            if str(label_row['code']) in pred_answer.translate(str.maketrans('', '', '()[]{}*&^%$#@!+-=_<>,.?/\\|"**')):
                                pred_label = label_row['code']
                                break
                            
                    # If no code found, check for name matches with word permutations
                    if pred_label is None:
                        for _, label_row in labels_dict.iterrows():
                            if isinstance(pred_answer, str):
                                name_words = set(label_row['name'].lower().translate(str.maketrans('', '', '()[]{}*&^%$#@!+-=_<>,.?/\\|"**')).split())
                                answer_words = set(pred_answer.lower().translate(str.maketrans('', '', '()[]{}*&^%$#@!+-=_<>,.?/\\|"**')).split())
                                if name_words.issubset(answer_words):
                                    pred_label = label_row['code']
                                    break
                
                    # If still no match found, use full answer
                    if pred_label is None:
                        pred_label = pred_answer
                        # print(f"No match found for {pred_answer}")
                else:
                    # Clean and split the predicted answer
                    if isinstance(pred_answer, str):
                        pred_words = set(pred_answer.lower().translate(str.maketrans('', '', '()[]{}*&^%$#@!+-=_<>,.?/\\|"**')).split())
                    else:
                        pred_words = set()
                    
                    # Check each ground truth label for word permutation match
                    found_match = False
                    for gtl in ground_truth_label_sets[gt_col]:
                        gtl_words = set(str(gtl).lower().translate(str.maketrans('', '', '()[]{}*&^%$#@!+-=_<>,.?/\\|"**')).split())
                        if gtl_words.issubset(pred_words):
                            pred_label = gtl
                            found_match = True
                            break
                    
                    if not found_match:
                        pred_label = pred_answer
                        # print(f"No match found for {pred_answer}")
                        
                # print(gt_label)
                # print(pred_label)
                
                # If predicted label not in ground truth labels, count as "not id"
                if pred_label not in ground_truth_label_sets[gt_col]:
                    conf_matrix.loc[(gt_col, gt_label), ('all', 'not id')] += 1
                else:
                    conf_matrix.loc[(gt_col, gt_label), (predicted_columns[i], pred_label)] += 1
        
        # Save confusion matrix to Excel
        conf_matrix.to_excel(output_path+'.xlsx')
        conf_matrix.to_csv(output_path+'.csv')
                
        return conf_matrix

def compute_score_metrics(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch, fold_number=None, save_overall_metrics=True):
    """
    Compute score metrics based on confusion matrix.
    
    Args:
        vlm_name (str): Name of the VLM model
        dataset_name (str): Name of the dataset
        task_type (str): Type of task (see get_global_info()['available_task_types'])
        reviewed (bool): Whether to use reviewed yes/no answers
        save_overall_metrics (bool, optional): Whether to save overall metrics for model comparison. Defaults to True.
        compute_confmat_from_scratch (bool, optional): Whether to recompute confusion matrix. Defaults to False.
        
    Returns:
        dict: Dictionary containing various score metrics computed from the confusion matrix
    """
    conf_matrix = compute_confusion_matrix(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch, fold_number=fold_number)

    # Calculate metrics for each class
    per_class_metrics = {}
    classes = conf_matrix.index
    
    total_samples = conf_matrix.values.sum()
    class_weights = {}
    
    for cls in classes:
        # Get true positives, false positives, true negatives, false negatives
        tp = conf_matrix.loc[cls, cls]
        fp = conf_matrix[cls].sum() - tp
        fn = conf_matrix.loc[cls].sum() - tp
        # tn = conf_matrix.values.sum() - tp - fp - fn
        
        # Calculate class weight
        class_weights[cls] = conf_matrix.loc[cls].sum() / total_samples
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        # npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        per_class_metrics[cls] = {
            'Precision (PPV)': precision,
            # 'NPV': npv,
            'Sensitivity (Recall)': sensitivity,
            # 'Specificity': specificity,
            'F1 Score': f1
        }
    # Convert metrics to DataFrame
    per_class_metrics_df = pd.DataFrame(per_class_metrics).T 

    # Calculate overall metrics
    total_tp = sum(conf_matrix.loc[cls, cls] for cls in classes)
    total_fp = sum(conf_matrix[cls].sum() - conf_matrix.loc[cls, cls] for cls in classes)
    total_fn = sum(conf_matrix.loc[cls].sum() - conf_matrix.loc[cls, cls] for cls in classes)

    # Calculate weighted F1 score
    weighted_f1 = sum(per_class_metrics[cls]['F1 Score'] * class_weights[cls] for cls in classes)

    overall_metrics = pd.DataFrame({
        'Precision (PPV)': [total_tp / (total_tp + total_fp)  if (total_tp + total_fp) > 0 else 0],
        'Sensitivity (Recall)': [total_tp / (total_tp + total_fn)  if (total_tp + total_fn) > 0 else 0],
        'F1 Score': [2 * total_tp / (2 * total_tp + total_fp + total_fn)  if (2 * total_tp + total_fp + total_fn) > 0 else 0],
        'Weighted F1 Score': [weighted_f1]
    }, index=['Overall'])
    
    metrics = {
        'per_class_metrics': per_class_metrics_df,
        'overall_metrics': overall_metrics
    }

    if save_overall_metrics:
        score_metrics_paths = get_score_metrics_paths(task_type, reviewed, fold_number=fold_number, file_type_extension=None)

        precision_score_path = score_metrics_paths['precision_score_path']
        # Check if precision scores file exists
        if os.path.exists(precision_score_path + '.csv'):
            precision_scores_df = pd.read_csv(precision_score_path + '.csv', index_col=0)
        elif os.path.exists(precision_score_path + '.xlsx'):
            precision_scores_df = pd.read_excel(precision_score_path + '.xlsx', index_col=0)
        else:
            if fold_number != None:
                precision_scores_df = pd.DataFrame(columns=[vlm_name],index=[dataset_name+f', fold {fold_number}'])
            else:
                precision_scores_df = pd.DataFrame(columns=[vlm_name], index=[dataset_name])        
        # Update precision score for current dataset and VLM
        if fold_number != None:
            precision_scores_df.loc[dataset_name+f', fold {fold_number}', vlm_name] = overall_metrics['Precision (PPV)'].values[0]
        else:
            precision_scores_df.loc[dataset_name, vlm_name] = overall_metrics['Precision (PPV)'].values[0]
        # Sort index and columns alphabetically
        precision_scores_df = precision_scores_df.sort_index()
        precision_scores_df = precision_scores_df.reindex(sorted(precision_scores_df.columns), axis=1)
        precision_scores_df.to_csv(precision_score_path + '.csv')
        precision_scores_df.to_excel(precision_score_path + '.xlsx')

       
        # Save sensitivity scores
        sensitivity_score_path = score_metrics_paths['sensitivity_score_path']
        if os.path.exists(sensitivity_score_path + '.csv'):
            sensitivity_scores_df = pd.read_csv(sensitivity_score_path + '.csv', index_col=0)
        elif os.path.exists(sensitivity_score_path + '.xlsx'):
            sensitivity_scores_df = pd.read_excel(sensitivity_score_path + '.xlsx', index_col=0)
        else:
            if fold_number != None:
                sensitivity_scores_df = pd.DataFrame(columns=[vlm_name], index=[dataset_name+f', fold {fold_number}'])
            else:
                sensitivity_scores_df = pd.DataFrame(columns=[vlm_name], index=[dataset_name])
        if fold_number != None:
            sensitivity_scores_df.loc[dataset_name+f', fold {fold_number}', vlm_name] = overall_metrics['Sensitivity (Recall)'].values[0]
        else:
            sensitivity_scores_df.loc[dataset_name, vlm_name] = overall_metrics['Sensitivity (Recall)'].values[0]
        # Sort index and columns alphabetically
        sensitivity_scores_df = sensitivity_scores_df.sort_index()
        sensitivity_scores_df = sensitivity_scores_df.reindex(sorted(sensitivity_scores_df.columns), axis=1)
        sensitivity_scores_df.to_csv(sensitivity_score_path + '.csv')
        sensitivity_scores_df.to_excel(sensitivity_score_path + '.xlsx')

      
        # Save F1 scores
        f1_score_path = score_metrics_paths['f1_score_path']
        if os.path.exists(f1_score_path + '.csv'):
            f1_scores_df = pd.read_csv(f1_score_path + '.csv', index_col=0)
        elif os.path.exists(f1_score_path + '.xlsx'):
            f1_scores_df = pd.read_excel(f1_score_path + '.xlsx', index_col=0)
        else:
            if fold_number != None:
                f1_scores_df = pd.DataFrame(columns=[vlm_name], index=[dataset_name+f', fold {fold_number}'])
            else:
                f1_scores_df = pd.DataFrame(columns=[vlm_name], index=[dataset_name])
        if fold_number != None:
            f1_scores_df.loc[dataset_name+f', fold {fold_number}', vlm_name] = overall_metrics['F1 Score'].values[0]
        else:
            f1_scores_df.loc[dataset_name, vlm_name] = overall_metrics['F1 Score'].values[0]
        # Sort index and columns alphabetically
        f1_scores_df = f1_scores_df.sort_index()
        f1_scores_df = f1_scores_df.reindex(sorted(f1_scores_df.columns), axis=1)
        f1_scores_df.to_csv(f1_score_path + '.csv')
        f1_scores_df.to_excel(f1_score_path + '.xlsx')

        # Save weighted F1 scores
        weighted_f1_score_path = score_metrics_paths['weighted_f1_score_path']
        if os.path.exists(weighted_f1_score_path + '.csv'):
            weighted_f1_scores_df = pd.read_csv(weighted_f1_score_path + '.csv', index_col=0)
        elif os.path.exists(weighted_f1_score_path + '.xlsx'):
            weighted_f1_scores_df = pd.read_excel(weighted_f1_score_path + '.xlsx', index_col=0)
        else:   
            if fold_number != None:
                weighted_f1_scores_df = pd.DataFrame(columns=[vlm_name], index=[dataset_name+f', fold {fold_number}'])
            else:
                weighted_f1_scores_df = pd.DataFrame(columns=[vlm_name], index=[dataset_name])
        if fold_number != None:
            weighted_f1_scores_df.loc[dataset_name+f', fold {fold_number}', vlm_name] = overall_metrics['Weighted F1 Score'].values[0]
        else:
            weighted_f1_scores_df.loc[dataset_name, vlm_name] = overall_metrics['Weighted F1 Score'].values[0]
        # Sort index and columns alphabetically
        weighted_f1_scores_df = weighted_f1_scores_df.sort_index()
        weighted_f1_scores_df = weighted_f1_scores_df.reindex(sorted(weighted_f1_scores_df.columns), axis=1)
        weighted_f1_scores_df.to_csv(weighted_f1_score_path + '.csv')
        weighted_f1_scores_df.to_excel(weighted_f1_score_path + '.xlsx')

    return metrics


def report_score_metrics(task_type, reviewed, file_type_extension='png'):
    """
    Create a report of all score metrics across VLMs and datasets.
    
    Args:
        reviewed (bool): Whether to use reviewed results ('yes') or not ('no')
        file_type_extension (str): Format to save the plot in ('pdf', 'png')
    """
    # Get paths to all score metric files
    score_metrics_paths = get_score_metrics_paths(task_type, reviewed)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis('off')
    
    # Add overall title
    plt.suptitle('Performance Metrics Across Models and Datasets for the' + task_type + ' task (for ' + ('reviewed' if reviewed == 'yes' else 'non-reviewed') + ' answers)', fontsize=14, y=0.98)
    
    # Initialize y position for tables
    y_pos = 1.0
    
    # Metrics to process and their display names
    metrics = {
        'precision_score_path': 'Precision Scores',
        # 'npv_score_path': 'Negative Predictive Value Scores',
        'sensitivity_score_path': 'Sensitivity Scores',
        # 'specificity_score_path': 'Specificity Scores',
        'f1_score_path': 'F1 Scores',
        'weighted_f1_score_path': 'Weighted F1 Scores'
    }
    
    for metric_path_key, metric_name in metrics.items():
        path = score_metrics_paths[metric_path_key]
        
        # Try to read the CSV first, then Excel if CSV not found
        try:
            if os.path.exists(path + '.csv'):
                df = pd.read_csv(path + '.csv', index_col=0)
            elif os.path.exists(path + '.xlsx'):
                df = pd.read_excel(path + '.xlsx', index_col=0)
            else:
                continue

            # Sort index and columns alphabetically
            df = df.sort_index()
            df = df.reindex(sorted(df.columns), axis=1)
                
            # Create table
            table = ax.table(
                cellText=df.round(3).values,
                rowLabels=df.index,
                colLabels=df.columns,
                cellLoc='center',
                loc='center',
                bbox=[0.1, y_pos-0.2, 0.8, 0.15]
            )
            
            # Add title for this metric
            ax.text(0.5, y_pos, metric_name, 
                   horizontalalignment='center',
                   fontsize=12,
                   fontweight='bold')
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Make row and column labels bold
            for cell in table._cells:
                if cell[0] == 0 or cell[1] == -1:  # Header row or first column
                    table._cells[cell].set_text_props(weight='bold')
            
            # Update y position for next table
            y_pos -= 0.25
            
        except Exception as e:
            print(f"Error processing {metric_name}: {str(e)}")
    
    # Adjust layout and save
    plt.tight_layout()
    

    score_metrics_report_path = get_score_metrics_report_path(task_type, reviewed, file_type_extension)
    
    # Save the figure
    plt.savefig(score_metrics_report_path, bbox_inches='tight', dpi=300)
    plt.close()



def plot_confusion_matrix(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch=True, file_type_extension='png'):
    """
    Plot confusion matrix and performance metrics.
    
    Args:
        vlm_name (str): Name of the vision language model
        dataset_name (str): Name of the dataset
        task_type (str): Type of task (see get_global_info()['available_task_types'])
        reviewed (bool): Whether to plot based on reviewed results or not
        compute_confmat_from_scratch (bool): Compute the confusion matrix from scratch or not (default: True)
        file_type_extension (str): Format to save the plot in ('pdf', 'png')
    """
    # Font size parameter for all text elements
    font_size = 20
    
    # Compute confusion matrix
    conf_matrix = compute_confusion_matrix(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch=compute_confmat_from_scratch)

    metrics = compute_score_metrics(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch=compute_confmat_from_scratch, save_overall_metrics=True)
    

    # Create figure with three subplots
    # Adjust figure size based on confusion matrix dimensions
    n_classes = len(conf_matrix)
    width = max(20, n_classes * 2)  # Scale width with number of classes, minimum 20
    height = max(15, n_classes * 1.5)     # Scale height with number of classes, minimum 10
    
    # Adjust width ratios - make confusion matrix wider for more classes
    conf_mat_ratio = max(1.5, n_classes/4)  # Scale ratio with classes, minimum 1.5
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(width, height), 
                                       gridspec_kw={'width_ratios': [conf_mat_ratio, 0.5, 1]})
    
    # Convert raw counts to percentages
    conf_matrix_pct = conf_matrix.astype(float).copy()
    for idx in conf_matrix.index:
        row_sum = conf_matrix.loc[idx].sum()
        if row_sum > 0:
            conf_matrix_pct.loc[idx] = conf_matrix.loc[idx] / row_sum * 100
    
    # Plot confusion matrix percentages
    im = ax1.imshow(conf_matrix_pct.values, cmap='Greys', vmin=0, vmax=100)
    
    # Add percentage labels
    for i in range(conf_matrix_pct.shape[0]):
        for j in range(conf_matrix_pct.shape[1]):
            text = f'{conf_matrix_pct.iloc[i, j]:.1f}'
            color_val = conf_matrix_pct.iloc[i, j] / 100
            text_color = 'white' if color_val > 0.3 else 'black'
            ax1.text(j, i, text, ha='center', va='center', color=text_color, fontsize=font_size)
    
    # Set labels
    ax1.set_xticks(np.arange(len(conf_matrix_pct.columns)))
    ax1.set_yticks(np.arange(len(conf_matrix_pct.index)))
    # Format MultiIndex labels
    xticklabels = [f'{col[0]}: {col[1]}' for col in conf_matrix_pct.columns]
    yticklabels = [f'{idx[0]}: {idx[1]}' for idx in conf_matrix_pct.index]
    ax1.set_xticklabels(xticklabels, rotation=90, fontsize=font_size)
    ax1.set_yticklabels(yticklabels, fontsize=font_size)
    ax1.set_xlabel('Predicted', fontsize=font_size)
    ax1.set_ylabel('True', fontsize=font_size)
    ax1.set_title('Confusion Matrix (%)', fontsize=font_size)
    
    # Plot ground truth counts
    gt_counts = conf_matrix.sum(axis=1)
    im2 = ax2.imshow(gt_counts.values.reshape(-1, 1), cmap='Greys')
    
    # Add count labels
    for i in range(len(gt_counts)):
        text = f'{gt_counts.iloc[i]:.0f}'
        color_val = gt_counts.iloc[i] / gt_counts.max()
        text_color = 'white' if color_val > 0.3 else 'black'
        ax2.text(0, i, text, ha='center', va='center', color=text_color, fontsize=font_size)
    
    # Set labels for ground truth counts
    ax2.set_yticks(np.arange(len(gt_counts)))
    ax2.set_yticklabels([f'{idx[0]}: {idx[1]}' for idx in gt_counts.index], fontsize=font_size)
    ax2.set_xticks([])
    ax2.set_title('Ground Truth\nCounts', fontsize=font_size)
    
    metrics_df = metrics['per_class_metrics']
    metrics_df = pd.concat([metrics_df, metrics['overall_metrics']])   
    
    # Plot metrics
    pos1 = ax1.get_position()
    height1 = pos1.height
    bottom1 = pos1.y0
    
    # Plot metrics as heatmap
    im = ax3.imshow(metrics_df.values, cmap='Greys', vmin=0, vmax=100)
    # Add percentage labels
    for i in range(metrics_df.shape[0]):
        for j in range(metrics_df.shape[1]):
            text = f'{metrics_df.values[i, j]:.2f}'
            color_val = metrics_df.values[i, j] / 100
            text_color = 'white' if color_val > 0.3 else 'black'
            ax3.text(j, i, text, ha='center', va='center', color=text_color, fontsize=font_size)
    
    # Set labels
    ax3.set_xticks(np.arange(len(metrics_df.columns)))
    ax3.set_yticks(np.arange(len(metrics_df.index)))
    ax3.set_xticklabels(metrics_df.columns, rotation=90, fontsize=font_size)
    ax3.set_yticklabels(metrics_df.index, fontsize=font_size)
    # Scale plot to match confusion matrix height
    pos3 = ax3.get_position()
    ax3.set_position([pos3.x0, bottom1, pos3.width, height1])
    
    ax3.set_title('Performance Metrics (%)', pad=20, fontsize=font_size)

    # Extract filename without extension for plot title
    if reviewed:    
        reviewed_yes_no_str = 'reviewed'
    elif not reviewed:
        reviewed_yes_no_str = 'not reviewed'

    plot_title = 'Confusion matrix and performance metrics for ' + dataset_name + ' dataset' + ' for the ' + task_type + ' task' + ' with ' + vlm_name + ' model' + ' (' + reviewed_yes_no_str + ')'
    fig.suptitle(plot_title, y=1.05, fontsize=font_size+2, fontweight='bold')
    
    # Adjust layout and save plot
    plt.tight_layout()
    conf_mat_plot_path = get_conf_mat_path(vlm_name, dataset_name, task_type, reviewed, file_type_extension=file_type_extension)['conf_mat_plot_path'] 
    plt.savefig(conf_mat_plot_path, bbox_inches='tight')
    plt.close()

# vlm_name = 'ft:gpt-4o-2024-08-06:personal:bonemarrow-n-pl-5:AuoO9vo2'
# dataset_name = 'Bone_Marrow_Cyto'
# task_type = '0shot_classification'
# compute_confmat_from_scratch = True
# for reviewed in [False,True]:
#     plot_confusion_matrix(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch=compute_confmat_from_scratch)

# compute_confmat_from_scratch = True
# dataset_name = 'Acevedo'
# task_type = '0shot_classification'
# reviewed = False

# vlms= ['gpt-4o',
#                     'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-1:AwdygYO3',
#                     'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-5:Awe2yBGI',
#                     'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-10:AweEXbUp',
#                     'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-25:AwfgA0tK',
#                     'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-50:Awfmw6Cp',
#                     'ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-100:AwoQsgDv']

# for vlm_name in vlms:
#     plot_confusion_matrix(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch=compute_confmat_from_scratch)

# report_score_metrics(task_type, reviewed, file_type_extension='png')

if __name__ == '__main__':
    compute_confmat_from_scratch = True
    global_info = get_global_info()
    for dataset_name in global_info['available_datasets']:
        # for vlm_family in global_info['available_model_families']:
        #     vlm_name = global_info['recommended_models'][vlm_family]
        for vlm_name in  global_info['available_models']:
            for task_type in [t for t in global_info['available_task_types'] if t != 'nonstructured']:
                for reviewed in [True, False]:
                    try:
                        plot_confusion_matrix(vlm_name, dataset_name, task_type, reviewed, compute_confmat_from_scratch=compute_confmat_from_scratch)
                    except Exception as e:
                        print(f"Error plotting confusion matrix for {vlm_name} on {dataset_name} for {task_type} with {reviewed} reviews: {e}")

    for task_type in [t for t in global_info['available_task_types'] if t != 'nonstructured']:
        for reviewed in [True, False]:
            try:
                report_score_metrics(task_type, reviewed, file_type_extension='png')
            except Exception as e:
                print(f"Error reporting score metrics for {task_type} with {reviewed} reviews: {e}")


