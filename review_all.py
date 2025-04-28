from dataset_info_and_paths import get_dataset_info, get_result_path, get_review_model, get_global_info 
from run_api_inquiry import run_api_review
import os   


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run review on all models for a data set')
    parser.add_argument('--dataset_name', type=str, choices=get_global_info()['available_datasets'],
                      help='Name of dataset to evaluate.') 

    args = parser.parse_args()


    global_info = get_global_info()

    available_models = global_info['available_models']
    available_task_types = global_info['available_task_types']
    # Remove 'nonstructured' from available task types
    available_task_types = [task_type for task_type in available_task_types if task_type != 'nonstructured']

    for task_type in ['nonstructured']: # available_task_types:
        print(f'Reviewing {task_type} for {args.dataset_name}')
        for vlm_name in  ['ft:gpt-4o-2024-08-06:marrlab-helmholtz-munich:acevedo-n-200:AxILVLLx']:  #available_models: # ['llama-3.2-multimodal-11B']:
            print(f'Reviewing {vlm_name} for {args.dataset_name} for {task_type}')
            nonreviewed_results_path = get_result_path(vlm_name, args.dataset_name, task_type, reviewed = False, file_type_extension='csv')['answers_path']
            reviewed_results_path = get_result_path(vlm_name, args.dataset_name, task_type, reviewed = True, file_type_extension='csv')['answers_path']

            if not os.path.exists(nonreviewed_results_path):
                print(f"File {nonreviewed_results_path} does not exist")
                continue

            if os.path.exists(reviewed_results_path):
                print(f"File {reviewed_results_path} already exists")
                continue

            review_model = get_review_model(vlm_name)

            print(f"Loaded files and running review for {vlm_name} on {args.dataset_name} for {task_type}. Reviewing with {review_model}")

            run_api_review(vlm_name, args.dataset_name, task_type)