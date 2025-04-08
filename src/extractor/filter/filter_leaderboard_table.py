import argparse
import json
import logging
import os

from copy import deepcopy
from tqdm import tqdm

import torch
import transformers
from datasets import Dataset, load_from_disk

from transformers import BitsAndBytesConfig


logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
)

MESSAGES_TEMPLATE = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "<|PROMPT|>"},
    ]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


def llm_process(prompt: str, pipeline: transformers.pipelines, max_new_tokens: int) -> str:
    
    messages = deepcopy(MESSAGES_TEMPLATE)
    if len(prompt) > 10000:
        prompt = prompt[:10000]
    messages[1]['content'] = messages[1]['content'].replace("<|PROMPT|>", prompt)

    outputs = pipeline(
        messages,
        max_new_tokens=max_new_tokens,
        pad_token_id = pipeline.tokenizer.eos_token_id,
        do_sample=False,
        top_p=1.0,
        num_beams=1,
        temperature=1.0,
    )
    response = outputs[0]["generated_text"][-1]['content']

    return response


def filter_table(dataset: Dataset, pipeline: transformers.pipeline, template_prompt: str) -> Dataset:

    fildered_dataset = []

    for instance in tqdm(dataset):
        tables_list = json.loads(instance['tables_list'])
        tables_index = json.loads(instance['tables_index'])
        
        filtered_index = []
        for i, table in enumerate(tables_list):
            table_src = table[0]
            prompt = deepcopy(template_prompt)
            prompt = prompt.replace("{Table LaTeX}", table_src)
            output = llm_process(prompt, pipeline, max_new_tokens=32)

            if 'true' in output.lower():
                filtered_index.append(i)
        
        for idx in filtered_index:
            filtered_instance = {
                'paper_id': instance['paper_id'],
                'full_tex': instance['full_paper_latex_code'],
                'table_source': tables_list[idx][0],
                'table_index': tables_index[idx],
            }
            fildered_dataset.append(filtered_instance)            

    logging.info(f"Filtered Dataset Length with Tables of Interest: {len(fildered_dataset)}")

    keys = fildered_dataset[0].keys()
    huggingface_dict = {}
    for k in keys:
        huggingface_dict[k] = [fildered_domain_instance[k] for fildered_domain_instance in fildered_dataset]
    filtered_dataset = Dataset.from_dict(huggingface_dict)
    
    return filtered_dataset


def main(ml_domain_table_ds: str, output_path: str, model_name_or_path: str, prompt_path: str, 
         cache_dir: str, shard_idx: int, total_shards: int):

    logging.info(f"Classifying whether the table contains the leaderboard table")
    
    dataset = load_from_disk(ml_domain_table_ds)

    logging.info(f"Processing shard idx {shard_idx+1} out of {total_shards} shards")
    logging.info(f"Filtered outputs will be saved in {output_path}")

    dataset = dataset.shard(num_shards=total_shards, index=shard_idx)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name_or_path,
        model_kwargs={"torch_dtype": torch.float16,
                      "device_map": "auto",
                      "quantization_config": bnb_config,
                      "cache_dir": cache_dir},
    )

    with open(prompt_path, 'r') as f:
        template_prompt = f.read().strip()

    filtered_dataset = filter_table(dataset, pipeline, template_prompt)    
    filtered_dataset.save_to_disk(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ml_table_ds', type=str)
    parser.add_argument('--ml_leaderboard_table_ds', type=str)
    parser.add_argument('--prompt_path', type=str, default="./prompt/filter_leaderboard.txt")
    parser.add_argument('--model_name_or_path', type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--total_shards', type=int, default=1)
    args = parser.parse_args()
    
    main(args.ml_table_ds, args.ml_leaderboard_table_ds, args.model_name_or_path, args.prompt_path,
         args.cache_dir, args.shard_idx, args.total_shards)
    logging.info("Finished filtering leaderboard tables")
