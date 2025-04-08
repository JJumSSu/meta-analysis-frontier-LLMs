import os
import logging
import argparse

import openai

from copy import deepcopy
from datasets import load_from_disk

from api.prompt import call_openai_model, get_tokens_and_price


logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
)


def main(args):
    deployment_name = None
    if args.api_source == 'azure':        
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        openai.api_base = args.openai_api_base
        openai.api_type = args.openai_api_type
        openai.api_version = args.openai_api_verion
        deployment_name = args.model_name_or_path
        
    dataset = load_from_disk(args.hf_ds_path)

    dataset_names = []
    for i in dataset:
        dataset_name = i['context_augmented_table_results_extracted']['dataset']
        dataset_names.append(dataset_name)

    subsets = []
    for i in dataset:
        subset = i['context_augmented_table_results_extracted']['subset']
        subsets.append(subset)

    assert len(dataset_names) == len(subsets)

    dataset_subset_concat = []
    for i in range(len(dataset_names)):
        if subsets[i] == "xx":
            dataset_subset_concat.append(dataset_names[i])
        else:
            dataset_subset_concat.append(dataset_names[i] + "<|SEP|>" + subsets[i])
    
    unique_datsets = list(set(dataset_subset_concat))
    
    with open(args.prompt_path, 'r') as f:
        prompt_template = f.read()

    prompt_list = []
    for i, instance in enumerate(unique_datsets):
        prompt = deepcopy(prompt_template)

        if "<|SEP|>" not in instance:
            prompt = prompt.replace("{{dataset}}", instance)
            prompt = prompt.replace("{{subset}}", "xx")
        else:
            dataset_name, subset = instance.split("<|SEP|>")
            prompt = prompt.replace("{{dataset}}", dataset_name)
            prompt = prompt.replace("{{subset}}", subset)
        
        prompt_list.append(prompt)

    results = call_openai_model(prompt_list, deployment_name, 
                                temperature=args.temperature, top_p=args.top_p, 
                                max_tokens=args.max_tokens,
                                sleep_time=2)

    get_tokens_and_price(prompt_list, results)
    
    unique_dataset_to_description = {}
    for i in range(len(unique_datsets)):
        unique_dataset_to_description[unique_datsets[i]] = results[i]

    dataset_descriptions = []
    description_source = []
    m = 0
    for dataset_subset in dataset_subset_concat:
        description = unique_dataset_to_description[dataset_subset]
        dataset_descriptions.append(description)
        if '<FAILED>' in description:
            m += 1
            description_source.append('<FAILED>')
        else:
            description_source.append('GPT4o')
    
    logging.info(f"Number of failed dataset descriptions: {m} out of {len(dataset_descriptions)}")

    dataset = dataset.add_column("dataset_description", dataset_descriptions)    
    dataset = dataset.add_column("description_source", description_source)
    
    logging.info(f"Saving {len(dataset)} dataset tables' descriptions")
    dataset.save_to_disk(args.hf_ds_output_path)

    logging.info(f"Generation Finished")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_source', type=str, choices=['azure', 'open_source'], default='azure')
    parser.add_argument('--backend', type=str, default='gpt-4o')
    parser.add_argument('--deployment_name', type=str)
    parser.add_argument('--openai_api_verion', type=str)
    parser.add_argument('--openai_api_type', type=str)
    parser.add_argument('--openai_api_base', type=str)
    
    parser.add_argument('--prompt_path', type=str, default='./extractor/generate_description/prompt/generate_description.txt')

    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)            
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--top_p', type=float, default=1.0)

    args = parser.parse_args()
    
    logging.info(f"Start Generating Datset Descriptions")

    main(args)
