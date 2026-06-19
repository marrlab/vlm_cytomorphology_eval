import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, Llama4ForConditionalGeneration, AutoProcessor


def load_llama_model(vlm_name='llama-3.2-multimodal-11B'):
    if vlm_name == 'llama-3.2-multimodal-11B':
        model_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/llama-3.2-multimodal-11B/llama-3.2-11B-model"
        processor_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/llama-3.2-multimodal-11B/llama-3.2-11B-processor"

    elif vlm_name == 'llama-3.2-multimodal-90B':
        model_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/llama-3.2-multimodal-90B/llama-3.2-90B-model"
        processor_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/llama-3.2-multimodal-90B/llama-3.2-90B-processor"

    elif vlm_name == 'Llama-4-Scout-17B-16E':
        model_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/Llama-4-Scout-17B-16E/checklist.chk"
        processor_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/Llama-4-Scout-17B-16E/tokenizer.model"

    elif vlm_name == 'Llama-4-Scout-17B-16E-Instruct':
        model_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/Llama-4-Scout-17B-16E-Instruct/checklist.chk"
        processor_path = "/lustre/groups/labs/marr/qscd01/workspace/llama_multimodal_models/Llama-4-Scout-17B-16E-Instruct/tokenizer.model"
    else:
        raise ValueError(f"Model {vlm_name} not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'llama-3.2' in vlm_name: 
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif 'Llama-4' in vlm_name:
        model = Llama4ForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flex_attention",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Model {vlm_name} not supported")

    model.tie_weights()

    processor = AutoProcessor.from_pretrained(processor_path)


    return model, processor


def llama_api_visual_inquiry(image_path, prompt_text, vlm_name='llama-3.2-multimodal-11B', **kwargs):

    model = kwargs.get('model')
    processor = kwargs.get('processor')
    max_new_tokens = kwargs.get('max_new_tokens', 10000)  # Default to 10000 if not provided

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(image_path)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = processor.decode(output[0])

    # full_output = processor.decode(output)

    # print(full_output)

    usage = 0 

    return answer, usage


def llama_api_text_inquiry(prompt_text, vlm_name='llama-3.2-multimodal-11B', **kwargs):
    
    
    # model = kwargs.get('model')
    # processor = kwargs.get('processor')
    # max_new_tokens = kwargs.get('max_new_tokens', 10000)  # Default to 10000 if not provided

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # image_path = '/lustre/groups/aih/ivan.kukuljan/VLMevaluation/Datasets/white_image_400x400.png'

    # image = Image.open(image_path)

    # messages = [
    #     {"role": "user", "content": [
    #         {"type": "image"},
    #         {"type": "text", "text": prompt_text}
    #     ]}
    # ]
    # input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    # inputs = processor(
    #     image,
    #     input_text,
    #     add_special_tokens=False,
    #     return_tensors="pt"
    # ).to(model.device)

    # output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # answer = processor.decode(output[0])

    # # full_output = processor.decode(output)

    # # print(full_output)

    # usage = 0 

    # return answer, usage
    
    
    model = kwargs.get('model')
    processor = kwargs.get('processor')
    max_new_tokens = kwargs.get('max_new_tokens', 10000)  # Default to 10000 if not provided
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        None,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    
    # inputs = processor(
    #     input_text,
    #     add_special_tokens=False,
    #     return_tensors="pt"
    # ).to(model.device)


    output = model.generate(**inputs, max_new_tokens=max_new_tokens) 

    answer = processor.decode(output[0])

    # full_output = processor.decode(output)

    # print(full_output)

    usage = 0 

    return answer, usage



def llama_multiimage_api_visual_inquiry(image_paths, prompt_texts, vlm_name='llama-3.2-multimodal-11B', **kwargs):

    if len(image_paths) != len(prompt_texts):
        raise ValueError("The number of image paths and prompt texts must be the same.")

    def prepare_messages(prompt_texts, image_paths):

        images = [Image.open(image_path) for image_path in image_paths]
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]        
        for prompt_text, image in zip(prompt_texts, images):
            messages[0]["content"].append({"type": "image", "image": image})
            messages[0]["content"].append({"type": "text", "text": prompt_text})
        
        return messages, images

    model = kwargs.get('model')
    processor = kwargs.get('processor')
    max_new_tokens = kwargs.get('max_new_tokens', 10000)  # Default to 10000 if not provided

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    messages, images = prepare_messages(prompt_texts, image_paths)

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        images,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = processor.decode(output[0])

    # full_output = processor.decode(output)

    # print(full_output)

    usage = 0 

    return answer, usage