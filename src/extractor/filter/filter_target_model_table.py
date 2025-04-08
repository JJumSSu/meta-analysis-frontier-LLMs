import argparse
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


def filter_table(dataset: Dataset, pipeline: transformers.pipeline, prompt_path: str,
                 model_name: str, model_keyword: str) -> Dataset:

    fildered_dataset = []

    with open(prompt_path, 'r') as f:
        template_prompt = f.read().strip()

    logging.info(f"Current Target Model: {model_name}")
    logging.info(f"Prompt: {template_prompt}")

    for instance in tqdm(dataset):
        table_src = instance['table_source']

        if model_keyword not in table_src.lower():
            continue
        
        prompt = template_prompt.replace("{Table LaTeX}", table_src)
        prompt = prompt.replace("{Model Name}", model_name)
        output = llm_process(prompt, pipeline, max_new_tokens=32)

        if 'true' in output.lower():
            fildered_dataset.append(instance)          

    logging.info(f"Filtered Tables of Target Model: {len(fildered_dataset)}")

    keys = fildered_dataset[0].keys()
    huggingface_dict = {}
    for k in keys:
        huggingface_dict[k] = [fildered_domain_instance[k] for fildered_domain_instance in fildered_dataset]
    filtered_dataset = Dataset.from_dict(huggingface_dict)
    
    return filtered_dataset


def main(ml_leaderboard_table_ds: str, output_path: str, model_name_or_path: str, prompt_path: str,
         model_keyword: str, model_name: str, cache_dir: str):
    
    logging.info(f"Classifying whether the table contains the target model's experimental results")
    
    dataset = load_from_disk(ml_leaderboard_table_ds)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name_or_path,
        model_kwargs={"torch_dtype": torch.float16,
                      "device_map": "auto",
                      "quantization_config": bnb_config,
                      "cache_dir": cache_dir},
    )

    filtered_dataset = filter_table(dataset, pipeline, prompt_path, model_name, model_keyword) 
    filtered_dataset.save_to_disk(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ml_leaderboard_table_ds', type=str)
    parser.add_argument('--target_model_leaderboard_ds', type=str)
    parser.add_argument('--model_name_or_path', type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument('--model_keyword', type=str, default="gpt")
    parser.add_argument('--model_name', type=str, default="GPT-4", choices=["GPT-4", "GPT-4o", "Gemini1.0 Pro", "Claude3 Opus"])
    parser.add_argument('--prompt_path', type=str)
    parser.add_argument('--cache_dir', type=str)
    args = parser.parse_args()
    
    main(args.ml_leaderboard_table_ds, args.target_model_leaderboard_ds, args.model_name_or_path, 
         args.prompt_path, args.model_keyword, args.model_name, args.cache_dir)
    logging.info("Finished classification")
