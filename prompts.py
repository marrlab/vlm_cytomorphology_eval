#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:48:41 2024

@author: ivan
"""

import pandas as pd

def get_all_classes(dataset_CSV_path: str):
    """
    Get all unique classes for each column in a given dataset.
    
    Args:
        dataset_CSV_path (str): Path to the CSV file describing the dataset
    
    Prints the unique classes for each column in the dataset.
    """
    df = pd.read_csv(dataset_CSV_path)

    for column in df.columns:
        print(column)
        print(df[column].unique())
    return 0


def get_prompt(dataset_name: str, task_type: str, reviewed=False):
    """
    Get the appropriate prompt text for a given dataset and structure type.
    
    Args:
        dataset_name (str): Name of the dataset to get prompt for (e.g. 'AML_Matek', 'Bone_Marrow_Cyto', 'WBCAtt')
        task_type (str): Which task to get prompt for (see get_global_info()['available_task_types'])
        reviewed (bool): If we are using a chatbot to review previous answers of a chatbot to this task
    
    Returns:
        dict: Dictionary containing prompt text for the specified dataset. 
        Dictionary keys refer to the column names in the dataset, and the values are the prompts for the respective columns.
    """

    prompt_AML_Matek_0shot_classification = {
        "cell_type": """Consider the input image. Take a moment to think. Consider what features do the cells in the image have. Which of the white blood cell types listed below is shown? 
    Write just the cell type and nothing else. Choose one of the possible labels provided below (exactly as written here):
    BAS Basophil
    EBO Erythroblast
    EOS Eosinophil
    KSC Smudge cell
    LYA Lymphocyte (atypical)
    LYT Lymphocyte (typical)
    MMZ Metamyelocyte
    MOB Monoblast
    MON Monocyte
    MYB Myelocyte
    MYO Myeloblast
    NGB Neutrophil (band)
    NGS Neutrophil (segmented)
    PMB Promyelocyte (bilobled)
    PMO Promyelocyte"""
    }

    prompt_AML_Matek_0shot_classification_review = {
        "cell_type": """Which of the classes listed below does the chatbot's answer regarding the cell type belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    BAS Basophil
    EBO Erythroblast
    EOS Eosinophil
    KSC Smudge cell
    LYA Lymphocyte (atypical)
    LYT Lymphocyte (typical)
    MMZ Metamyelocyte
    MOB Monoblast
    MON Monocyte
    MYB Myelocyte
    MYO Myeloblast
    NGB Neutrophil (band)
    NGS Neutrophil (segmented)
    PMB Promyelocyte (bilobled)
    PMO Promyelocyte"""
    }
        
    prompt_AML_Matek_nonstructured = {'cell_type': 'Consider the input image. Take a moment to think. Consider what features do the cells in the image have. Which type of white blood cell is shown? Write just the cell type and nothing else.'}

    prompt_Bone_Marrow_Cyto_0shot_classification = {'cell_type': """Consider the input image. Take a moment to think. Consider what features do the cells in the image have. Which of the white blood cell types listed below is shown? 
    Write your full considerations but conclude your reply with 'Answer: ' and then write one of the possible labels provided below (exactly as written here):
    ABE 	Abnormal eosinophil
    ART 	Artefact
    BAS 	Basophil
    BLA 	Blast
    EBO 	Erythroblast
    EOS 	Eosinophil
    FGC 	Faggott cell
    HAC 	Hairy cell
    KSC 	Smudge cell
    LYI 	Immature lymphocyte
    LYT 	Lymphocyte
    MMZ 	Metamyelocyte
    MON 	Monocyte
    MYB 	Myelocyte
    NGB 	Band neutrophil
    NGS 	Segmented neutrophil
    OTH 	Other cell
    PEB 	Proerythroblast
    PLM 	Plasma cell
    PMO 	Promyelocyte"""}

    prompt_Bone_Marrow_Cyto_0shot_classification_review = {'cell_type': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the cell type belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    ABE 	Abnormal eosinophil
    ART 	Artefact
    BAS 	Basophil
    BLA 	Blast
    EBO 	Erythroblast
    EOS 	Eosinophil
    FGC 	Faggott cell
    HAC 	Hairy cell
    KSC 	Smudge cell
    LYI 	Immature lymphocyte
    LYT 	Lymphocyte
    MMZ 	Metamyelocyte
    MON 	Monocyte
    MYB 	Myelocyte
    NGB 	Band neutrophil
    NGS 	Segmented neutrophil
    OTH 	Other cell
    PEB 	Proerythroblast
    PLM 	Plasma cell
    PMO 	Promyelocyte"""}

    prompt_Bone_Marrow_Cyto_nonstructured = {'cell_type': 'Consider the input image. Take a moment to think. Consider what features do the cells in the image have. Which of the white blood cell types listed below is shown? Write just the cell type and nothing else.'}

    prompt_Acevedo_0shot_classification = {
        "cell_type": """Consider the input image. Take a moment to think. Consider what features do the cells in the image have. Which of the white blood cell types listed below is shown? 
    Write just the cell type and nothing else. Choose one of the possible labels provided below (exactly as written here):
    Band Neutrophil
    Basophil
    Eosinophil
    Erythroblast
    Lymphocyte
    Metamyelocyte
    Monocyte
    Myelocyte
    Platelet
    Promyelocyte
    Segmented Neutrophil"""
    }

    prompt_Acevedo_0shot_classification_review = {
        "cell_type": """Which of the classes listed below does the chatbot's answer regarding the cell type belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    Band Neutrophil
    Basophil
    Eosinophil
    Erythroblast
    Lymphocyte
    Metamyelocyte
    Monocyte
    Myelocyte
    Platelet
    Promyelocyte
    Segmented Neutrophil"""
    }
        
    prompt_Acevedo_explainability = {
        "cell_type": """Consider the input image. Take a moment to think. Consider what features do the cells in the image have. Which of the white blood cell types listed below is shown? 
    Write just the cell type and nothing else. Choose one of the possible labels provided below (exactly as written here):
    Band Neutrophil
    Basophil
    Eosinophil
    Erythroblast
    Lymphocyte
    Metamyelocyte
    Monocyte
    Myelocyte
    Platelet
    Promyelocyte
    Segmented Neutrophil
    
    Now consider the cell features listed below. Think how much each of them contributed to your cell classification decision that you made above. 
    Next to each feature, write an importance score how much the feature was important for you classification decision. The scores should be float numbers. All the scores together should sum to 100.
    
    Cell Shape 
    Cell Size
    Nuclear Shape 
    Nuclear Segmentation 
    Nuclear-to-Cytoplasmic Ratio 
    Nuclear Membrane Appearance 
    Nucleoli 
    Chromatin Pattern     
    Cytoplasmic Volume 
    Cytoplasmic Color 
    Cytoplasmic Border
    Granule Presence 
    Granule Type      
    Inclusions (Presence of Auer rods, Döhle bodies, or other cytoplasmic inclusions)    
    Cytoplasmic Basophilia 
    Erythrocytes
    Platelets, Thrombocytes
    Surrounding of the cell 
    Technical properties of the image (resolution, light, noise, etc.)    

    Are there any other features that you consider important for the classification decision? If yes, write them below.
    """
    }
        
    prompt_Acevedo_pixel_level_explainability = {
        "cell_type": """Consider the input image. Take a moment to think. Consider what features do the cells in the image have. Which of the white blood cell types listed below is shown? 
    Write just the cell type and nothing else. Choose one of the possible labels provided below (exactly as written here):
    Band Neutrophil
    Basophil
    Eosinophil
    Erythroblast
    Lymphocyte
    Metamyelocyte
    Monocyte
    Myelocyte
    Platelet
    Promyelocyte
    Segmented Neutrophil
    
    Now plot an image highlighting which pixels in the image that we gave you were the most crucial for the classification decision you made. Paint very important pixels in red and pixels with intermediate importance in yellow.
    """
    }

    features_Acevedo_explainability_review = [
    'Cell Shape ',
    'Cell Size',
    'Nuclear Shape ',
    'Nuclear Segmentation ',
    'Nuclear-to-Cytoplasmic Ratio ',
    'Nuclear Membrane Appearance ',
    'Nucleoli ',
    'Chromatin Pattern ',
    'Cytoplasmic Volume ',
    'Cytoplasmic Color ',
    'Cytoplasmic Border',
    'Granule Presence ',
    'Granule Type ',
    'Inclusions (Presence of Auer rods, Döhle bodies, or other cytoplasmic inclusions) ',
    'Cytoplasmic Basophilia ',
    'Erythrocytes',
    'Platelets, Thrombocytes',
    'Surrounding of the cell ',
    'Technical properties of the image (resolution, light, noise, etc.) '
    ]

        
    prompt_Acevedo_nonstructured = {'cell_type': """Consider the input image. Take a moment to think. Consider what features do the cells in the image have. Which of the white blood cell types listed below is shown? 
    Choose one of the possible labels provided below (exactly as written here):
    Band Neutrophil
    Basophil
    Eosinophil
    Erythroblast
    Lymphocyte
    Metamyelocyte
    Monocyte
    Myelocyte
    Platelet
    Promyelocyte
    Segmented Neutrophil
    
    Explain in detail your decision and the reasoning that lead you to the decision. Which parts of the cells and features did you consider? 
    
    How certain are you about your classification? Which other labels could be correct? Why did you choose this label in the end?
    """
    }


    prompt_WBCAtt_0shot_classification = {'label': """Consider the input image. Take a moment to think. Consider what features do the cells in the image have. Which of the white blood cell types listed below is shown? Write just the cell type and nothing else. Choose one of the possible labels provided below (exactly as written here):
    Neutrophil
    Eosinophil
    Basophil
    Lymphocyte
    Monocyte
    """,
    'cell_size': """Consider the white blood cell shown in the input image. Take a moment to think. What is the size of the white blood cell? Choose one of the possible labels provided below (exactly as written here):
    big
    small
    """,
    'cell_shape': """Consider the white blood cell shown in the input image. Take a moment to think. What is the shape of the white blood cell? Choose one of the possible labels provided below (exactly as written here):
    round
    irregular
    """,
    'nucleus_shape': """Consider the white blood cell shown in the input image. Take a moment to think. What is the shape of the nucleus of the white blood cell? Choose one of the possible labels provided below (exactly as written here):
    unsegmented-band
    unsegmented-round
    segmented-multilobed
    segmented-bilobed
    irregular
    unsegmented-indented
    """,
    'nuclear_cytoplasmic_ratio': """Consider the white blood cell shown in the input image. Take a moment to think. What is the ratio of the nucleus to the cytoplasm of the white blood cell? Choose one of the possible labels provided below (exactly as written here):
    low
    high
    """,
    'chromatin_density': """Consider the white blood cell shown in the input image. Take a moment to think. What is the chromatin density of the white blood cell? Choose one of the possible labels provided below (exactly as written here):
    densely
    loosely
    """,
    'cytoplasm_vacuole': """Consider the white blood cell shown in the input image. Take a moment to think. Does the white blood cell have a cytoplasmic vacuole? Choose one of the possible labels provided below (exactly as written here):
    no
    yes
    """,   
    'cytoplasm_texture': """Consider the white blood cell shown in the input image. Take a moment to think. What is the texture of the cytoplasm of the white blood cell? Choose one of the possible labels provided below (exactly as written here):
    clear
    frosted
    """,
    'cytoplasm_colour': """Consider the white blood cell shown in the input image. Take a moment to think. What is the colour of the cytoplasm of the white blood cell? Choose one of the possible labels provided below (exactly as written here):
    light blue
    blue
    purple blue
    """,
    'granule_type': """Consider the white blood cell shown in the input image. Take a moment to think. What is the type of the granules in the white blood cell? Choose one of the possible labels provided below (exactly as written here):
    small
    round
    coarse
    nil
    """,
    'granule_colour': """Consider the white blood cell shown in the input image. Take a moment to think. What is the colour of the granules in the white blood cell? Choose one of the possible labels provided below (exactly as written here):
    pink
    purple
    red
    nil
    """,
    'granularity': """Consider the white blood cell shown in the input image. Take a moment to think. Does the white blood cell have granules? Choose one of the possible labels provided below (exactly as written here):
    yes
    no
    """,
    }


    prompt_WBCAtt_0shot_classification_review = {'label': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the cell type belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    Neutrophil
    Eosinophil
    Basophil
    Lymphocyte
    Monocyte
    """,
    'cell_size': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the cell size belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    big
    small
    """,
    'cell_shape': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the cell shape belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    round
    irregular
    """,
    'nucleus_shape': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the nucleus shape belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    unsegmented-band
    unsegmented-round
    segmented-multilobed
    segmented-bilobed
    irregular
    unsegmented-indented
    """,
    'nuclear_cytoplasmic_ratio': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the nuclear cytoplasmic ratio belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    low
    high
    """,
    'chromatin_density': """Which of the classes listed below does the chatbot's answer regarding the chromatin density belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    densely
    loosely
    """,
    'cytoplasm_vacuole': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the cytoplasmic vacuole belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    no
    yes
    """,   
    'cytoplasm_texture': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the cytoplasmic texture belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    clear
    frosted
    """,
    'cytoplasm_colour': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the cytoplasmic colour belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    light blue
    blue
    purple blue
    """,
    'granule_type': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the granule type belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    small
    round
    coarse
    nil
    """,
    'granule_colour': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the granule colour belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    pink
    purple
    red
    nil
    """,
    'granularity': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the granule presence belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    yes
    no
    """,
    }

    prompt_HiCervix_0shot_classification = {'cell_type': """Consider the input image. Take a moment to think. Consider what features do cells in the image have. Which of the  types listed below is shown? 
    Write just the cell type and nothing else. Choose one of the possible labels provided below (exactly as written here):
    Normal cell
    Endocervical cell
    Repair cell
    Metaplasia cell
    Pseudokoilocytes by glycogen    
    Atrophy
    Endometrial cell
    Hyperchromatic crowded groups
    Atypical squamous cells of undetermined significance
    Low-grade squamous intraepithelial lesion
    Atypical squamous cells, cannot exclude high-grade squamous intraepithelial lesion
    High-grade squamous intraepithelial lesion
    Squamous cell carcinoma
    Atypical glandular cell
    Atypical glandular cell- not otherwise specified
    Atypical glandular cell- favor neoplastic
    Adenocarcinoma
    Atypical glandular cell (endocervical cell)- not otherwise specified
    Atypical glandular cell (endometrial cell)- not otherwise specified
    Adenocarcinoma of endocervical cell
    Adenocarcinoma of endometrial cell
    Fungal organisms morphologically consistent with Candida spp.
    Bacteria morphologically consistent with Actinomyces spp.
    Trichomonas vaginalis
    Cellular changes consistent with herpes simplex virus
    Coccobacilli/Shift in flora suggestive of bacterial vaginosis """,
    }

    prompt_HiCervix_0shot_classification_review = {'cell_type': """Consider only the chatbot's answer and ignore any potentially attached images. Which of the classes listed below does the chatbot's answer regarding the cell type belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    Normal cell
    Endocervical cell
    Repair cell
    Metaplasia cell
    Pseudokoilocytes by glycogen    
    Atrophy
    Endometrial cell
    Hyperchromatic crowded groups
    Atypical squamous cells of undetermined significance
    Low-grade squamous intraepithelial lesion
    Atypical squamous cells, cannot exclude high-grade squamous intraepithelial lesion
    High-grade squamous intraepithelial lesion
    Squamous cell carcinoma
    Atypical glandular cell
    Atypical glandular cell- not otherwise specified
    Atypical glandular cell- favor neoplastic
    Adenocarcinoma
    Atypical glandular cell (endocervical cell)- not otherwise specified
    Atypical glandular cell (endometrial cell)- not otherwise specified
    Adenocarcinoma of endocervical cell
    Adenocarcinoma of endometrial cell
    Fungal organisms morphologically consistent with Candida spp.
    Bacteria morphologically consistent with Actinomyces spp.
    Trichomonas vaginalis
    Cellular changes consistent with herpes simplex virus
    Coccobacilli/Shift in flora suggestive of bacterial vaginosis""", }
    
    prompt_MLL23_0shot_classification = {
        "cell_type": """Consider the input image. Take a moment to think. Consider what features do the cells in the image have. Which of the white blood cell types listed below is shown? 
    Write just the cell type and nothing else. Choose one of the possible labels provided below (exactly as written here):
    band neutrophil
    basophil
    eosinophil
    erythroblast
    lymphocyte
    metamyelocyte
    monocyte
    myelocyte
    platelet
    promyelocyte
    segmented neutrophil"""
    }

    prompt_MLL23_0shot_classification_review = {
        "cell_type": """Which of the classes listed below does the chatbot's answer regarding the cell type belong to? Write just the label (exacly as written below) and nothing else:
    NA (Chatbot is unsure/ambiguious/doesn't know/no answer provided/class cannot be determined)
    band neutrophil
    basophil
    eosinophil
    erythroblast
    lymphocyte
    metamyelocyte
    monocyte
    myelocyte
    platelet
    promyelocyte
    segmented neutrophil"""
    }
        
    prompt_MLL23_nonstructured = {'cell_type': 'Consider the input image. Take a moment to think. Consider what features do the cells in the image have. Which type of white blood cell is shown? Write just the cell type and nothing else.'}

    if dataset_name == 'AML_Matek':
        if task_type == '0shot_classification':
            if reviewed==False:
                return prompt_AML_Matek_0shot_classification
            elif reviewed==True:
                return prompt_AML_Matek_0shot_classification_review
        elif task_type == 'nonstructured':
            return prompt_AML_Matek_nonstructured


    elif 'Bone_Marrow_Cyto' in dataset_name:
        if task_type == '0shot_classification':
            if reviewed==False:
                return prompt_Bone_Marrow_Cyto_0shot_classification
            elif reviewed==True:
                return prompt_Bone_Marrow_Cyto_0shot_classification_review
        elif task_type == 'nonstructured':
            return prompt_Bone_Marrow_Cyto_nonstructured

    elif dataset_name == 'Acevedo':
        if task_type == '0shot_classification':
            if reviewed==False:
                return prompt_Acevedo_0shot_classification
            elif reviewed==True:
                return prompt_Acevedo_0shot_classification_review
        elif task_type == 'nonstructured':
            return prompt_Acevedo_nonstructured
        elif task_type == 'explainability':
            if reviewed==False:
                return prompt_Acevedo_explainability
            elif reviewed==True:
                return features_Acevedo_explainability_review

    elif dataset_name == 'WBCAtt':
        if task_type == '0shot_classification':
            if reviewed==False:
                return prompt_WBCAtt_0shot_classification
            elif reviewed==True:
                return prompt_WBCAtt_0shot_classification_review
        elif task_type == 'nonstructured':
            return None

    elif dataset_name == 'HiCervix':
        if task_type == '0shot_classification':
            if reviewed==False:
                return prompt_HiCervix_0shot_classification
            elif reviewed==True:
                return prompt_HiCervix_0shot_classification_review
        elif task_type == 'nonstructured':
            return None
        
    elif dataset_name == 'MLL23':
        if task_type == '0shot_classification':
            if reviewed==False:
                return prompt_MLL23_0shot_classification
            elif reviewed==True:
                return prompt_MLL23_0shot_classification_review
        elif task_type == 'nonstructured':
            return None

    else:
        raise ValueError(f"Task type {task_type} not found for dataset {dataset_name}. Check prompts.py and add if needed.")

# dataset_name='WBCAtt'
# task_type='review'
# prompt_dict=get_prompt(dataset_name, task_type)

# print(prompt_dict)


