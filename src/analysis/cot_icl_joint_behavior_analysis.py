import argparse
import pandas as pd
import os
import json

from collections import defaultdict

from datasets import load_dataset

from utils import process_ds


def get_cot_stanard_delta_for_zero_shot_and_few_shot(total_dataset, args):
    
    total_delta_dict = defaultdict(list)
    
    for comparison_group in range(2):

        if comparison_group == 0:
            dataset = total_dataset.filter(lambda x: x['number_of_shots'] != 0)
        else:
            dataset = total_dataset.filter(lambda x: x['number_of_shots'] == 0)

        dataset_names = dataset['dataset']
        model_names = dataset['model_name']
        subset = dataset['subset']
        number_of_shots = dataset['number_of_shots']
        prompting_method = dataset['prompting_method']
        source_arxiv_id = dataset['table_source_arxiv_id']
        dataset_descriptions = dataset['dataset_description']
        metric = dataset['metric']
        metric_value = dataset['metric_value']
        table_index = dataset['table_index']

        identifiers = []
        for mn, sa, dn, s, ns, m, ti in zip(model_names, source_arxiv_id, dataset_names, subset, number_of_shots, metric, table_index):
            identifier = "_".join([mn, sa, dn, s, str(ns), m, str(ti)])
            identifiers.append(identifier)
        
        identifier_to_prompting_methods = defaultdict(list)
        for i, identifier in enumerate(identifiers):
            identifier_to_prompting_methods[identifier].append((prompting_method[i], metric_value[i], source_arxiv_id[i], 
                                                                dataset_names[i], dataset_descriptions[i], number_of_shots[i]))

        filtered_identifier_to_prompting_methods = {k: v for k, v in identifier_to_prompting_methods.items() if len(v) > 1}

        refiltered_identifier_to_prompting_methods = defaultdict(list) # pre-filter
        for k in filtered_identifier_to_prompting_methods.keys():
            to_include = False
            for t in filtered_identifier_to_prompting_methods[k]:
                if 'cot' in t[0].lower():
                    if 'no' not in t[0].lower():
                        to_include = True
                elif 'chain' in t[0].lower() and 'thought' in t[0].lower():
                    to_include = True
            if to_include:
                refiltered_identifier_to_prompting_methods[k] = filtered_identifier_to_prompting_methods[k]

        cot_prompt_filter_list = []
        with open(args.cot_prompt_filtering_list_path, "r") as f:
            for i in f:
                cot_prompt_filter_list.append(i.strip().lower())            

        standard_prompt_filter_list = []
        with open(args.standard_prompt_filtering_list_path, "r") as f:
            for i in f:
                standard_prompt_filter_list.append(i.strip().lower())

        total_deltas = []
        total_papers = []
        for k in refiltered_identifier_to_prompting_methods.keys():
            prompting_methods_and_values = refiltered_identifier_to_prompting_methods[k]
            
            perf_standard_prompts = []
            perf_cot_prompts = []
            standard_prompts = []
            cot_prompts = []

            for prompt in prompting_methods_and_values:
                
                is_cot_prompt = False
                if 'cot' in prompt[0].lower():
                    if 'no' not in prompt[0].lower():
                        is_cot_prompt = True
                elif 'chain' in prompt[0].lower() and 'thought' in prompt[0].lower():
                    is_cot_prompt = True
                    
                if is_cot_prompt:
                    if prompt[0].lower() not in cot_prompt_filter_list:
                        perf_cot_prompts.append(prompt[1])
                        cot_prompts.append(prompt[0])
                else:
                    if prompt[0].lower() not in standard_prompt_filter_list:
                        perf_standard_prompts.append(prompt[1])
                        standard_prompts.append(prompt[0])            
            
            if len(cot_prompts) > 0 and len(standard_prompts) > 0:
                perf_cot_prompts = sum(perf_cot_prompts) / len(perf_cot_prompts)
                perf_standard_prompts = sum(perf_standard_prompts) / len(perf_standard_prompts)
                delta = perf_cot_prompts - perf_standard_prompts
                total_deltas.append(delta)
                total_papers.append(prompt[2])
    
        if comparison_group == 0:
            total_delta_dict['few_shot'] = total_deltas
        else:
            total_delta_dict['zero_shot'] = total_deltas
        
    for k in total_delta_dict.keys():
        print("CoT - Standard Prompting on", k)
        print(f"Median: {pd.Series(total_delta_dict[k]).median():.2f}")
        print(f"Mean: {pd.Series(total_delta_dict[k]).mean():.2f}")
        print(f"Q1: {pd.Series(total_delta_dict[k]).quantile(0.25):.2f}")
        print(f"Q3: {pd.Series(total_delta_dict[k]).quantile(0.75):.2f}")
        print(f"Std: {pd.Series(total_delta_dict[k]).std():.2f}")
        total_delta_dict[k] = [round(i, 2) for i in total_delta_dict[k]]

    return total_delta_dict


def get_cot_delta_for_zero_shot_and_few_shot(total_dataset, args):
    
    dataset = total_dataset
    total_delta_dict= defaultdict(list)
        
    dataset_names = dataset['dataset']
    model_names = dataset['model_name']
    subset = dataset['subset']
    number_of_shots = dataset['number_of_shots']
    prompting_method = dataset['prompting_method']
    source_arxiv_id = dataset['table_source_arxiv_id']
    metric = dataset['metric']
    metric_value = dataset['metric_value']
    table_index = dataset['table_index']

    cot_prompt_filter_list = []
    with open(args.cot_prompt_filtering_list_path, "r") as f:
        for i in f:
            cot_prompt_filter_list.append(i.strip().lower())  

    identifiers = []
    for mn, sa, dn, s, p, m, ti in zip(model_names, source_arxiv_id, dataset_names, subset, prompting_method, metric, table_index):
        identifier = "_".join([mn, sa, dn, s, p, m, str(ti)])
        identifiers.append(identifier)

    shots_conditioned_on_identifier = defaultdict(list)
    for i, identifier in enumerate(identifiers):
        shots_conditioned_on_identifier[identifier].append((number_of_shots[i], metric_value[i], prompting_method[i], source_arxiv_id[i]))
    
    shots_conditioned_on_identifier = {k: v for k, v in shots_conditioned_on_identifier.items() if len(v) > 1}

    filtered_identifier_to_prompting_methods = defaultdict(list)
    for k in shots_conditioned_on_identifier.keys():
        to_include = False
        for t in shots_conditioned_on_identifier[k]:
            if 'cot' in t[2].lower():
                if 'no' not in t[2].lower():
                    if t[2].lower() not in cot_prompt_filter_list:
                        to_include = True
            elif 'chain' in t[2].lower() and 'thought' in t[2].lower():
                if t[2].lower() not in cot_prompt_filter_list:
                    to_include = True
        if to_include:
            filtered_identifier_to_prompting_methods[k] = shots_conditioned_on_identifier[k]          

    total_deltas_more_shot_less_shot = []
    total_deltas_zero_shot_few_shot = []

    total_papers = []

    for k in filtered_identifier_to_prompting_methods.keys(): # prompting method is same, but has variations in few-shot numbers       
    
        prompting_methods_and_values = filtered_identifier_to_prompting_methods[k]
        prompting_methods_and_values.sort(key=lambda x: x[0])
        prompting_methods_and_values = [r for r in prompting_methods_and_values if r[0] < 100]
        
        if len(prompting_methods_and_values) < 2:
            continue
        
        total_papers.append(prompting_methods_and_values[0][3])

        for instance in prompting_methods_and_values: # zero-shot versus few-shot
            if str(instance[0]) == '0':
                is_pass = True
            
        if is_pass:
            do_not_append = False
            zero_shot_performance = [instance[1] for instance in prompting_methods_and_values if str(instance[0]) == '0']
            if len(zero_shot_performance) != 1: # two conflicting results from different papers
                do_not_append = True
            else:
                zero_shot_performance = zero_shot_performance[0]
                few_shot_performance = []
                for instance in prompting_methods_and_values:
                    if str(instance[0]) == '0':
                        continue
                    else:
                        assert int(instance[0]) != 0
                        assert int(instance[0]) > 0
                        few_shot_performance.append(instance[1])
                
                few_shot_performance = sum(few_shot_performance) / len(few_shot_performance)
                delta = few_shot_performance - zero_shot_performance
                if not do_not_append:
                    total_deltas_zero_shot_few_shot.append(delta)

        deltas = []
        for i in range(len(prompting_methods_and_values)):
            for j in range(i+1, len(prompting_methods_and_values)):
                if prompting_methods_and_values[j][0] == prompting_methods_and_values[i][0]:
                    continue
                cur_delta = prompting_methods_and_values[j][1] - prompting_methods_and_values[i][1]
                assert int(prompting_methods_and_values[j][0]) - int(prompting_methods_and_values[i][0]) != 0
                assert int(prompting_methods_and_values[j][0]) - int(prompting_methods_and_values[i][0]) > 0
                deltas.append(cur_delta)

        if len(deltas) > 0:
            delta = sum(deltas) / len(deltas)
            total_deltas_more_shot_less_shot.append(delta)

    total_delta_dict['delta_zero_shot_few_shot'] = total_deltas_zero_shot_few_shot
    total_delta_dict['delta_more_shot_less_shot'] = total_deltas_more_shot_less_shot       

    for k in total_delta_dict.keys():
        print("CoT Prompting on", k)
        print(f"Median: {pd.Series(total_delta_dict[k]).median():.2f}")
        print(f"Mean: {pd.Series(total_delta_dict[k]).mean():.2f}")
        print(f"Q1: {pd.Series(total_delta_dict[k]).quantile(0.25):.2f}")
        print(f"Q3: {pd.Series(total_delta_dict[k]).quantile(0.75):.2f}")
        print(f"Std: {pd.Series(total_delta_dict[k]).std():.2f}")
        total_delta_dict[k] = [round(i, 2) for i in total_delta_dict[k]] 

    return total_delta_dict


def main(args):
    
    dataset = load_dataset(args.hf_ds_path)['train']
    dataset = process_ds(dataset, args.arxiv_id_to_venue, args.venue_filter, process_few_shot=True)
    
    total_cot_stadard_delta = get_cot_stanard_delta_for_zero_shot_and_few_shot(dataset, args)
    total_cot_delta_for_zero_shot_and_few_shot = get_cot_delta_for_zero_shot_and_few_shot(dataset, args)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Write the CoT standard delta data to a JSON file
    cot_standard_delta_path = os.path.join(args.output_dir, 'cot_standard_delta.json')
    with open(cot_standard_delta_path, 'w') as f:
        json.dump(total_cot_stadard_delta, f, indent=4)

    # Write the CoT delta for zero-shot and few-shot to a JSON file
    cot_delta_path = os.path.join(args.output_dir, 'cot_delta_zero_few_shot.json')
    with open(cot_delta_path, 'w') as f:
        json.dump(total_cot_delta_for_zero_shot_and_few_shot, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_ds_path', type=str, default='jungsoopark/LLMs-Performance-Data')
    parser.add_argument('--arxiv_id_to_venue', type=str, default='./metadata/arxiv_id_to_venue.json')
    parser.add_argument('--venue_filter', action='store_true', default=False)
    parser.add_argument('--cot_prompt_filtering_list_path', type=str, default='./classification_results/cot_prompt_filtering_list.txt')
    parser.add_argument('--standard_prompt_filtering_list_path', type=str, default='./classification_results/standard_prompt_filtering_list.txt')
    parser.add_argument('--output_dir', type=str, default='./analysis_outputs/')
    args = parser.parse_args()
    main(args)
