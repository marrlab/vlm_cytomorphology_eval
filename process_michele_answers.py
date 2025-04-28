import pandas as pd
import os
import numpy as np







def compute_average_score(answers_path):

    answers_df = pd.read_excel(answers_path)


    answers_correctly_predicted = answers_df[answers_df['ground_truth_cell_type'] == answers_df['predicted_cell_type']]

    answers_incorrectly_predicted = answers_df[answers_df['ground_truth_cell_type'] != answers_df['predicted_cell_type']]
    
    number_correctly_predicted = len(answers_correctly_predicted)
    number_incorrectly_predicted = len(answers_incorrectly_predicted)

    average_score_correctly_predicted = np.mean(answers_correctly_predicted['correctness (1-excellent, 2-good, 3-ok, 4-poor, 5-completely wrong)']).item()
    std_score_correctly_predicted = np.std(answers_correctly_predicted['correctness (1-excellent, 2-good, 3-ok, 4-poor, 5-completely wrong)']).item()
    average_score_incorrectly_predicted = np.mean(answers_incorrectly_predicted['correctness (1-excellent, 2-good, 3-ok, 4-poor, 5-completely wrong)']).item()
    std_score_incorrectly_predicted = np.std(answers_incorrectly_predicted['correctness (1-excellent, 2-good, 3-ok, 4-poor, 5-completely wrong)']).item()

    scores = {}
    scores['number_correctly_predicted'] = number_correctly_predicted
    scores['average_score_correctly_predicted'] = average_score_correctly_predicted
    scores['std_score_correctly_predicted'] = std_score_correctly_predicted
    scores['number_incorrectly_predicted'] = number_incorrectly_predicted
    scores['average_score_incorrectly_predicted'] = average_score_incorrectly_predicted
    scores['std_score_incorrectly_predicted'] = std_score_incorrectly_predicted
    
    
    return scores



def compute_michele_confusion_matrix(answers_path):

    answers_df = pd.read_excel(answers_path)

    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('Band Neutrophil ', 'Band Neutrophil')
    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('Band neutrophil', 'Band Neutrophil')

    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('Basophil ', 'Basophil')
    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('Eosinophil ', 'Eosinophil')
    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('Platelet ', 'Platelet')

    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('Eosinophile', 'Eosinophil')

    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('Segmented Neutrophil ', 'Segmented Neutrophil')
    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('band neutrophil', 'Band Neutrophil')
    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('destroyed cell ', 'Destroyed Cell')
    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('thrombocyte', 'Thrombocyte')
    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('Myelocyte  ', 'Myelocyte')

    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('band Neutrophil', 'Band Neutrophil')
    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('band neutrophil ', 'Band Neutrophil')
    answers_df['Michele_cell_type'] = answers_df['Michele_cell_type'].replace('segmented Neutrophil ', 'Segmented Neutrophil')

    confusion_matrix = pd.crosstab(answers_df['ground_truth_cell_type'], answers_df['Michele_cell_type'], 
                                   rownames=['True'], colnames=['Predicted'], margins=True) 


    # Get the directory of answers_path
    output_dir = os.path.dirname(answers_path)
    
    # Create output filename by appending '_confusion_matrix.xlsx' to the base name
    base_name = os.path.splitext(os.path.basename(answers_path))[0]
    output_path = os.path.join(output_dir, 'michele_answers_confusion_matrix.xlsx')
    
    # Save confusion matrix to Excel
    confusion_matrix.to_excel(output_path)
    print(f"Saved confusion matrix to {output_path}")
    
    return confusion_matrix


def compute_conf_mat_scores(conf_mat):

    # Remove 'All' row and column if present
    if 'All' in conf_mat.index:
        conf_mat = conf_mat.drop('All', axis=0)
    if 'All' in conf_mat.columns:
        conf_mat = conf_mat.drop('All', axis=1)

    print(conf_mat.columns)
    print(conf_mat.index)

    classes = list(conf_mat.index)
    # Remove 'All' from classes list if present
    #if 'All' in classes:
    #    classes.remove('All')

    # Compute the scores for the confusion matrix
    # The scores are the sum of the true positives, true negatives, false positives, and false negatives
    # The scores are normalized by the total number of samples 

    # Calculate overall metrics
    total_tp = sum(conf_mat.loc[cls, cls] for cls in classes)
    total_fp = sum(conf_mat[cls].sum() - conf_mat.loc[cls, cls] for cls in classes)
    total_fn = sum(conf_mat.loc[cls].sum() - conf_mat.loc[cls, cls] for cls in classes)

    overall_metrics = {
        'Precision (PPV)': [total_tp / (total_tp + total_fp)  if (total_tp + total_fp) > 0 else 0],
        'Sensitivity (Recall)': [total_tp / (total_tp + total_fn)  if (total_tp + total_fn) > 0 else 0],
        'F1 Score': [2 * total_tp / (2 * total_tp + total_fp + total_fn)  if (2 * total_tp + total_fp + total_fn) > 0 else 0],
    }

    return overall_metrics


        
    
    
    



answers_path = '/home/ivan/Helmholtz/VLMevaluation/Datasets/Acevedo/Michele_ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-200:AxILVLLx/us_complete_data.xlsx'

scores = compute_average_score(answers_path)

print(scores)

confusion_matrix = compute_michele_confusion_matrix(answers_path)
print(confusion_matrix)

overall_metrics = compute_conf_mat_scores(confusion_matrix)

print(overall_metrics)