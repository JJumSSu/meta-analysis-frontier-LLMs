import argparse
import json
import logging
import os

from copy import deepcopy

import tiktoken
import openai

from datasets import Dataset, load_from_disk

from api.prompt import call_openai_model, get_tokens_and_price

logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
)


def filter_metric(dataset: Dataset, deployment_name:str, prompt_template) -> Dataset:

    filtered_dataset = []
    adjusted_values = []
    standardized_metric_names = []

    prompt_list = []
    for instance in dataset:
        record = instance['context_augmented_table_results_extracted']
        metric = record['metric']
        value = record['value']

        if metric is None or value is None:
            metric = "fluency"
            value = "0"

        prompt = deepcopy(prompt_template)
        prompt = prompt.replace("{{METRIC_NAME}}", metric)
        prompt = prompt.replace("{{METRIC_VALUE}}", value)

        prompt_list.append(prompt)
    
    results = call_openai_model(prompt_list, deployment_name, 
                                temperature=args.temperature, top_p=args.top_p, 
                                max_tokens=args.max_tokens,
                                sleep_time=3)

    get_tokens_and_price(prompt_list, results)

    assert len(results) == len(dataset)

    for i, output in enumerate(results):
        
        output = output.replace('\n', '')
        output = output.replace('```json', '')
        output = output.replace('```', '')

        if '<FAILED>' in output:
            continue

        try:
            output = output.replace("'", '"')
            output = json.loads(output)
        except Exception:
            logging.info("parsing error, skipping...")
            continue

        try:
            output['Metric_Value'] = float(output['Metric_Value'])
        except Exception:
            logging.info("format error, skipping...")
            continue

        adjusted_values.append(output['Metric_Value'])
        standardized_metric_names.append(output['Metric_Name'])
        filtered_dataset.append(dataset[i])

    logging.info(f"Filtered Dataset Length of Records with Valid Metrics: {len(filtered_dataset)}")

    keys = filtered_dataset[0].keys()
    huggingface_dict = {}
    for k in keys:
        huggingface_dict[k] = [fildered_domain_instance[k] for fildered_domain_instance in filtered_dataset]
    filtered_dataset = Dataset.from_dict(huggingface_dict)
    filtered_dataset = filtered_dataset.add_column("adjusted_metric_value", adjusted_values)
    filtered_dataset = filtered_dataset.add_column("standardized_metric", standardized_metric_names)
    
    return filtered_dataset


def main(args):

    deployment_name = None
    if args.api_source == 'azure':        
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        openai.api_base = args.openai_api_base
        openai.api_type = args.openai_api_type
        openai.api_version = args.openai_api_verion
        deployment_name = args.deployment_name

    logging.info(f"Classifying whether the record contains the valid metrics")
    
    dataset = load_from_disk(args.hf_ds_path)

    logging.info(f"Current Dataset Length: {len(dataset)}")

    with open(args.prompt_path, 'r') as f:
        prompt_template = f.read()

    filtered_dataset = filter_metric(dataset, deployment_name, prompt_template)    
    filtered_dataset.save_to_disk(args.hf_ds_output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_source', type=str, choices=['azure', 'open_source'], default='azure')
    parser.add_argument('--backend', type=str, default='gpt-4o')
    parser.add_argument('--deployment_name', type=str)
    parser.add_argument('--openai_api_verion', type=str)
    parser.add_argument('--openai_api_type', type=str)
    parser.add_argument('--openai_api_base', type=str)

    parser.add_argument('--prompt_path', type=str, default='./extractor/extract/prompt/filter_metric.txt')

    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)

    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--top_p', type=float, default=1.0)
    
    args = parser.parse_args()
    
    main(args)
    logging.info("Finished classification")
