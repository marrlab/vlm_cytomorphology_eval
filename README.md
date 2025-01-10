# Evaluation of VLM models on cytology tasks and datasets

This is a repository for automated evaluation of VLM models on cytology tasks and datasets.

**Important: Read the entire README before you start working on the project, using and modifying the code!**

**If you are developing the code, inform the whole project group in Slack, what you intend to change. Ask Ivan for permission to change the logic of the code or the folder structure and naming conventions.**

**If you have any questions, ask Ivan or someone else in the project team.**

## Contents

- dataset_info_and_paths.py: contains functions that return paths to datasets, results folders, etc. This file defines the entire folder structure of the project and naming conventions. All the file names have to be defined here in order to avoid conflicts. In other files, you are not allowed to define any names - you are only allowed to call get_..._path functions from here.
- prepare_dataset.py: contains functions that samples a smaller subset of a large data set for the VLM evaluation. Use this for every new dataset that you're adding. Automatically creates also label csv files. Potential abbreviation_dictionary files have to be created manually.
- prompts.py: contains a function that returns the suitable prompt for the chosen task
- run_api_inquiry.py: contains function that runs the VLM models API inquiries for different tasks. Also contains a function that runs API enquiry to review the VLM models answers
- clean_llama_answers.py: contains function that cleans the llama answers. They come out weirdly structured and you need to run this function to clean them up.
- plot_conf_mat.py: contains function that plots the confusion matrix and generates a report

Subfolder vlm_models contains the VLM models API inquiry codes for each model.

## Generates the following folder structure in root_folder_path (defined in dataset_info_and_paths.py):

- root_folder    
    - Datasets
        - dataset_name
            dataset_name_labels.csv
            dataset_name_labels.xlsx
            dataset_name_abbreviation_dictionary.csv
            ...
            images
            ...
    - Results
        overview results across all datasets and models
        - dataset_name
            - answers
                answers provided by VLMs
            - confusion_matrices
                confusion matrices for each model
    - Plots  
        overview report plots across all datasets and models
        - dataset_name
            dataset specific plots

## Requirements

See requirements.txt

If by chance any of the requirements are missing in the file, add them please and inform Ivan.

## Usage

### Running the API call of a model

Run the API call for a dataset, task and model by running run_api_inquiry.py and parsing the arguments.

**Important: Before running the comercial VLMs, you will need to register and get an API key.**
Before running run_api_inquiry.py in the terminal, you'll need to export the API keys. For example:
```
export GEMINI_API_KEY = 'your_gemini_api_key'
export OPENAI_API_KEY = 'your_openai_api_key'
```
Don't share your key with anyone unless you would like them to burn your money. :) 

### Reporting results

To compute and plot confusion mattrices for all the models and datasets run plot_conf_mat.py. This also collects all the results and plots reports. 
If you want to plot only for a specific model or dataset, call individual funstions from plot_conf_mat.py.

### Adding a new dataset

To add a new dataset for evaluation:

1. Add dataset info to `dataset_info_and_paths.py`:
    - Add the dataset name to the available_datasets list in the get_global_info function
    - Follow the parameter descriptions and the pattern of previous datasets to the following info to the get_dataset_info function:
        - paths to original dataset and annotations
        - column names for subset sampling and image paths
        - which classes/labels to include
        - any dataset-specific configuration
        - ...

2. Create prompts in `prompts.py`:
   - Add dataset-specific prompts for all the tasks
   - Ensure prompts match the dataset's labels

3. Generate the evaluation subset:
   - Choose n_per_label - the number of samples per class
   - Run: `python prepare_dataset.py --dataset_name <name> --n_per_label <n>`
   This will automatically choose a subset of the entire dataset such that all the classes are equally represented. It will copy it to the vlm_eval_subset_folder_path and create a labels file. You should run this function only once for a new dataset. The function will not overwrite the existing subset. If you want to change the subset, you need to delete the existing subset folder and run the function again. If the subset was created by someone else, you need to ask everyone in the project team for a permission to delete it.

### Adding a new VLM model family

To add a new VLM model family for evaluation:

1. Add the model family to available_model_families list. Add the name of the most recommended model within the family to the recommended_models dictionary in the get_global_info function in dataset_info_and_paths.py

2. Add the model to the get_review_model function in dataset_info_and_paths.py - specify which model to use for reviewing the 
answers generated by the VLM model you're adding. If you believe that the model is strong enough, use itself, otherwise use gpt-4o.

3. Add a new api inquiry code file to the vlm_models folder on github. Call it <<model_family_name>>_api.py.
It needs to contain the following functions:
    - <<model_family_name>>_api_visual_inquiry. Takes image_path, prompt_text, vlm_name as inputs and returns the answer and token usage.This function is used to pass a single image and prompt to the VLM model.
    ...
    - <<model_family_name>>_api_task for every task type that requires more than a single image. It needs to take suitable inputs and return the answer and token usage. See other models for examples.
    ...
    - <<model_family_name>>_api_text_inquiry. Takes prompt_text, vlm_name as inputs and returns the answer and token usage. This function is used to review the answers generated by the VLM models.

See gpt_api.py for an example.

4. Add the model to the import_model function in run_api_inquiry.py. Specify possible model particularities in kwargs.

### Adding a new task

1. Add the task name to the available_task_types list in the get_global_info function in dataset_info_and_paths.py
2. Add the prompts for the task type for all datasets and for both reviewed and unreviewed cases to the prompts.py file.
3. Write the function run_api_task_type for the task in run_api_inquiry.py.
4. Add potential new task-specific functions to the model files in the vlm_models folder.
5. Add potential new task-specific functions to the import_model function in the run_api_inquiry.py file.







