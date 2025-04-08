import os
import logging
import argparse
import re

import tiktoken
import openai

from copy import deepcopy
from datasets import load_from_disk

from api.prompt import call_openai_model, get_tokens_and_price
from extractor.preprocess.tex.process_tex import preprocess_tex_src

tokenizer = tiktoken.get_encoding("o200k_base")

logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
)

def clean_arxiv_tex(tex_content):  
    tex_content = re.sub(r'\\appendix.*?\\end{document}', '', tex_content, flags=re.DOTALL)  
    tex_content = re.sub(r'\\begin{figure}.*?\\end{figure}', '', tex_content, flags=re.DOTALL)  
    tex_content = re.sub(r'%.*?\n', '', tex_content)   # Remove comments
    tex_content = re.sub(r'\s+', ' ', tex_content)   # Remove extra whitespace
    tex_content = re.sub(r'\\begin{thebibliography}.*?\\end{thebibliography}', '', tex_content, flags=re.DOTALL)  # Remove bibliography
    tex_content = re.sub(r'\\documentclass.*?\\begin{document}', '\\begin{document}', tex_content, flags=re.DOTALL)  # Remove packages
    tex_content = re.sub(r'\\(usepackage|documentclass|title|author|date|maketitle).*?\n', '', tex_content)  # Remove certain commands  
    tex_content = re.sub(r'\\cite[t|p]?{.*?}', '', tex_content)  # Remove citations
    tex_content = re.sub(r'\\begin{equation}.*?\\end{equation}', '', tex_content, flags=re.DOTALL) # Remove equations
    tex_content = re.sub(r'\\begin{align}.*?\\end{align}', '', tex_content, flags=re.DOTALL)  # Remove equations
    tex_content = re.sub(r'\\begin{table}.*?\\end{table}', '', tex_content, flags=re.DOTALL)

    return tex_content  


def truncate_length(prompt_list):
    new_prompt_list = []
    for prompt in prompt_list:
        prompt = prompt.replace("<|endofprompt|>", "")
        prompt = prompt.replace("<|endoftext|>", "")

        temp = tokenizer.encode(prompt)
        if len(temp) > 30000:
            temp = temp[:30000]
            truncated_prompt = tokenizer.decode(temp)
        else:
            truncated_prompt = prompt
        new_prompt_list.append(truncated_prompt)
    return new_prompt_list


def main(args):

    logging.info(args)

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

    dataset_subset_concats = []
    for i in range(len(dataset_names)):
        if subsets[i] == "xx":
            dataset_subset_concats.append(dataset_names[i])
        else:
            dataset_subset_concats.append(dataset_names[i] + "<|SEP|>" + subsets[i])

    dataset_descriptions = dataset['dataset_description']
    paper_ids = dataset['paper_id']
    citation_tags = dataset['dataset_link']

    dataset_info = list(zip(dataset_subset_concats, dataset_descriptions, paper_ids, citation_tags))
    
    target_dataset_info = []
    for dataset_subset, dataset_description, paper_id, citation_tag in dataset_info:
        if dataset_description == "<FAILED>":
            target_dataset_info.append((dataset_subset, dataset_description, paper_id, citation_tag))

    target_dataset_info = list(set(target_dataset_info))

    with open(args.prompt_path, 'r') as f:
        prompt_template = f.read()

    prompt_list = []
    source_list = []
    for i, (dataset_subset, dataset_description, paper_id, citation_tag) in enumerate(target_dataset_info):

        orig_source_path = args.orig_source_path
        citation_source_path = args.citation_source_path

        if len(citation_tag) > 0:
            source_path = os.path.join(citation_source_path, citation_tag)
            source_list.append(citation_tag)
        else:
            source_path = os.path.join(orig_source_path, paper_id)
            source_list.append(paper_id)
        
        tex_content = preprocess_tex_src(source_path)
        tex_content = clean_arxiv_tex(tex_content)

        prompt = deepcopy(prompt_template)
        
        if "<|SEP|>" not in dataset_subset:
            dataset_name = dataset_subset
            subset = "xx"
        else:
            dataset_name, subset = dataset_subset.split("<|SEP|>")

        prompt = prompt.replace("{{dataset}}", dataset_name)
        prompt = prompt.replace("{{subset}}", subset)
        prompt = prompt.replace("{{text_source}}", tex_content)
        prompt_list.append(prompt)

    prompt_list = truncate_length(prompt_list)

    assert len(prompt_list) == len(source_list)

    results = call_openai_model(prompt_list, deployment_name, 
                                temperature=args.temperature, top_p=args.top_p, 
                                max_tokens=args.max_tokens,
                                sleep_time=6)
    
    get_tokens_and_price(prompt_list, results)

    unique_dataset_subset_to_description = {}
    for i in range(len(target_dataset_info)):
        unique_dataset_subset_to_description[target_dataset_info[i][0]] = (results[i], source_list[i])

    dataset_descriptions_augmented = deepcopy(dataset['dataset_description'])
    dataset_description_source = deepcopy(dataset['description_source'])
    
    m = 0
    n = 0

    for i, instance in enumerate(dataset_subset_concats):
        if instance in unique_dataset_subset_to_description.keys():
            dataset_description = unique_dataset_subset_to_description[instance][0]
            dataset_descriptions_augmented[i] = dataset_description
            dataset_description_source[i] = unique_dataset_subset_to_description[instance][1]
            if '<FAILED>' in dataset_description:
                m += 1
            else:
                n += 1
            
    logging.info(f"Number of failed dataset extractions: {m} out of {m + n}")

    dataset = dataset.remove_columns('dataset_description')
    dataset = dataset.remove_columns('description_source')
    dataset = dataset.add_column('dataset_description', dataset_descriptions_augmented)
    dataset = dataset.add_column('description_source', dataset_description_source)
    
    logging.info(f"saving {len(dataset)} dataset descriptions")
    dataset.save_to_disk(args.hf_ds_output_path)

    logging.info(f"Extraction Generation Finished")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_source', type=str, choices=['azure', 'open_source'], default='azure')
    parser.add_argument('--backend', type=str, default='gpt-4o')
    parser.add_argument('--deployment_name', type=str)
    parser.add_argument('--openai_api_verion', type=str)
    parser.add_argument('--openai_api_type', type=str)
    parser.add_argument('--openai_api_base', type=str)

    parser.add_argument('--prompt_path', type=str, default='./extractor/generate_description/prompt/extract_description.txt')

    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)
    parser.add_argument('--orig_source_path', type=str)
    parser.add_argument('--citation_source_path', type=str)
            
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--top_p', type=float, default=1.0)
    args = parser.parse_args()
    
    logging.info(f"Start Extracting Dataset Descriptions")

    main(args)
