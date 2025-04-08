import os
import logging
import argparse
import re  
import json

from copy import deepcopy

import tiktoken
import openai

from datasets import load_from_disk

from api.prompt import call_openai_model, get_tokens_and_price

tokenizer = tiktoken.get_encoding("o200k_base")

logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
)


def parse_extracted_results(response: str):
    json_objects = re.findall(r'\{.*?\}', response)
    list_of_dicts = []
    for item in json_objects:
        try:
            temp = json.loads(item)
            list_of_dicts.append(temp)
        except Exception: 
            continue

    for dict_item in list_of_dicts:
        for key in dict_item:
            dict_item[key] = str(dict_item[key])
            
    return list_of_dicts
  

def clean_arxiv_tex_for_contextual_augmentation(tex_content):  
    tex_content = re.sub(r'\\appendix.*?\\end{document}', '', tex_content, flags=re.DOTALL)  
    tex_content = re.sub(r'\\begin{figure}.*?\\end{figure}', '', tex_content, flags=re.DOTALL)  
    tex_content = re.sub(r'%.*?\n', '', tex_content)   # Remove comments
    tex_content = re.sub(r'\s+', ' ', tex_content)   # Remove extra whitespace
    tex_content = re.sub(r'\\begin{thebibliography}.*?\\end{thebibliography}', '', tex_content, flags=re.DOTALL)  # Remove bibliography
    tex_content = re.sub(r'\\documentclass.*?\\begin{document}', '\\begin{document}', tex_content, flags=re.DOTALL)  # Remove packages
    tex_content = re.sub(r'\\(usepackage|documentclass|title|author|date|maketitle).*?\n', '', tex_content)  # Remove certain commands  
    tex_content = re.sub(r'\\begin{equation}.*?\\end{equation}', '', tex_content, flags=re.DOTALL) # Remove equations
    tex_content = re.sub(r'\\begin{align}.*?\\end{align}', '', tex_content, flags=re.DOTALL)  # Remove equations
    
    return tex_content  


def truncate_length(prompt_list):
    new_prompt_list = []
    for prompt in prompt_list:
        prompt = prompt.replace("<|endofprompt|>", "")
        prompt = prompt.replace("<|endoftext|>", "")

        temp = tokenizer.encode(prompt)
        if len(temp) > 40000:
            temp = temp[:40000]
            truncated_prompt = tokenizer.decode(temp)
        else:
            truncated_prompt = prompt
        new_prompt_list.append(truncated_prompt)
    return new_prompt_list


def main(args):
    deployment_name = None
    if args.api_source == 'azure':        
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        openai.api_base = args.openai_api_base
        openai.api_type = args.openai_api_type
        openai.api_version = args.openai_api_verion
        deployment_name = args.deployment_name
        
    dataset = load_from_disk(args.hf_ds_path)

    full_tex_contents = []

    for i in dataset:
        full_tex_contents.append(clean_arxiv_tex_for_contextual_augmentation(i['full_tex']))

    with open(args.prompt_path, 'r') as f:
        prompt_template = f.read()

    prompt_list = []

    for i, instance in enumerate(dataset):
        prompt = deepcopy(prompt_template)
        prompt = prompt.replace("{{records}}", instance['table_results_extracted'])
        prompt = prompt.replace("{{table_code}}", instance['table_source'])
        prompt = prompt.replace("{{text}}", full_tex_contents[i])
        prompt_list.append(prompt)

    prompt_list = truncate_length(prompt_list)
    results = call_openai_model(prompt_list, deployment_name, 
                                temperature=args.temperature, top_p=args.top_p, 
                                max_tokens=args.max_tokens,
                                sleep_time=6)

    get_tokens_and_price(prompt_list, results)

    parsed_results = []
    for result in results:
        parsed_result = parse_extracted_results(result)
        parsed_results.append(parsed_result)    

    dataset = dataset.add_column("context_augmented_table_results_extracted", parsed_results)    

    logging.info(f"flattening dataset")

    parsed_results = dataset['context_augmented_table_results_extracted']
    flattened_dataset = {}
    flattened_results = []
    for column in dataset.column_names:
        if column != 'context_augmented_table_results_extracted':
            flattened_dataset[column] = []
    
    for idx, result_list in enumerate(parsed_results):
        num_results = len(result_list)
        flattened_results.extend(result_list)
        
        for column in dataset.column_names:
            if column != 'context_augmented_table_results_extracted':
                flattened_dataset[column].extend([dataset[column][idx]] * num_results)
    
    flattened_dataset['context_augmented_table_results_extracted'] = flattened_results
    dataset = dataset.from_dict(flattened_dataset)

    logging.info(f"saving {len(dataset)} context augemented records")
    dataset.save_to_disk(args.hf_ds_output_path)

    logging.info(f"Context Augmentation Finished")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_source', type=str, choices=['azure', 'open_source'], default='azure')
    parser.add_argument('--backend', type=str, default='gpt-4o')
    parser.add_argument('--deployment_name', type=str)
    parser.add_argument('--openai_api_verion', type=str)
    parser.add_argument('--openai_api_type', type=str)
    parser.add_argument('--openai_api_base', type=str)

    parser.add_argument('--prompt_path', type=str, default='./extractor/extract/prompt/context_augment.txt')

    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)
            
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--top_p', type=float, default=1.0)
    args = parser.parse_args()
    
    logging.info(f"Start Augmenting Context for Extracted Records using Paper Tex")

    main(args)
