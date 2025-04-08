import logging
import time
import os

from typing import List
from tqdm import tqdm

import openai
import tiktoken

input_token_cost = 2.50 / 1_000_000
output_token_cost = 10.00 / 1_000_000
tokenizer = tiktoken.get_encoding("o200k_base")

logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
)

def call_openai_model(prompt_list: List[str], deployment_name: str, 
                                        temperature=0.001, top_p=1.0, max_tokens=8192, sleep_time=0):
    response_list = []
    for prompt in tqdm(prompt_list): 
        num_trials = 0
        while True:
            try:
                messages = [{"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(engine=deployment_name, messages=messages, 
                                                        temperature=temperature, top_p=top_p, 
                                                        max_tokens=max_tokens, seed=42)

                response = response["choices"][0]["message"]["content"]
                response_list.append(response)
                break
                
            except Exception as e:
                print(e)
                time.sleep(60)
                num_trials += 1
                if num_trials > 5:
                    raise ValueError(f"Failed to get response after 5 trials")

        time.sleep(sleep_time)
                   
    return response_list


def get_tokens_and_price(prompts_list, outputs_list):
    
    assert len(prompts_list) == len(outputs_list)

    total_price = 0
    total_input_price = 0
    total_output_price = 0
    avg_num_input_tokens = 0
    avg_num_output_tokens = 0
    total_number = 0
    
    for i, prompt in enumerate(prompts_list):
        try:
            num_input_tokens = len(tokenizer.encode(prompt))
            output = outputs_list[i]
            num_output_tokens = len(tokenizer.encode(output))

            avg_num_input_tokens += num_input_tokens
            avg_num_output_tokens += num_output_tokens

            instance_input_price = num_input_tokens * input_token_cost
            total_input_price += instance_input_price
            instance_output_price = num_output_tokens * output_token_cost
            total_output_price += instance_output_price
            total_number += 1 
        except Exception as e:
            logging.error(f"Error: {e}")
            continue

    total_price += total_input_price + total_output_price

    logging.info(f"Avg Num Input Tokens: {avg_num_input_tokens/total_number}")
    logging.info(f"Avg Num Output Tokens: {avg_num_output_tokens/total_number}")
    logging.info(f"Total Input Price: {total_input_price}")
    logging.info(f"Total Output Price: {total_output_price}")
    logging.info(f"Total Price: {total_price}")
