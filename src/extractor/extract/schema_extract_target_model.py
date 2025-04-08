import os
import logging
import argparse

from copy import deepcopy

import openai
import tiktoken
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
        deployment_name = args.deployment_name
        
    dataset = load_from_disk(args.hf_ds_path)

    with open(args.prompt_path, 'r') as f:
        prompt_template = f.read()

    prompt_list = []

    for instance in dataset:
        prompt = deepcopy(prompt_template)
        prompt = prompt.replace("{{target_model_name}}", args.target_model_name)
        prompt = prompt.replace("{{table_code}}", instance['table_source'])
        prompt_list.append(prompt)

    results = call_openai_model(prompt_list, deployment_name, 
                                temperature=args.temperature, top_p=args.top_p, 
                                max_tokens=args.max_tokens,
                                sleep_time=0)

    get_tokens_and_price(prompt_list, results)

    n = 0
    idx = []
    for i, result in enumerate(results):
        if "<failed>" in result.lower():
            n += 1
            idx.append(i)
    logging.info(f"{n} Failed out of {len(results)} Tables")

    dataset = dataset.add_column("table_results_extracted", results)    
    dataset = dataset.select([i for i in range(len(dataset)) if i not in idx])

    logging.info(f"saving {len(dataset)} Tables")
    dataset.save_to_disk(args.hf_ds_output_path)

    logging.info(f"Extraction Finished")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_source', type=str, choices=['azure', 'open_source'], default='azure')
    parser.add_argument('--backend', type=str, default='gpt-4o')
    parser.add_argument('--deployment_name', type=str)
    parser.add_argument('--openai_api_verion', type=str)
    parser.add_argument('--openai_api_type', type=str)
    parser.add_argument('--openai_api_base', type=str)

    parser.add_argument('--prompt_path', type=str, default='./extractor/extract/prompt/schema_extract_target_model.txt')
    parser.add_argument('--target_model_name', type=str, default='GPT-4')

    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)

    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--top_p', type=float, default=1.0)
    
    args = parser.parse_args()
    
    logging.info(f"Start Extracting Tables of Target Model Experiments")

    main(args)
