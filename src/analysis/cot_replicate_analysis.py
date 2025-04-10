import argparse
import json
import numpy as np
import os

from collections import defaultdict

from datasets import load_dataset

from utils import process_ds, create_boxplots


CATEGORIES = ['Tool Use', 'Multimodality', 'Math', 'Coding', 'Instruction Following', 'Safety', 'Knowledge', 'Reasoning', 'Multilinguality']
CATEGORIES_SPRAGUE = ['Entailment', 'Text classification', 'Generation', 'Mixed datasets', 'Encyclopedic knowledge', 'Context-aware QA', 'Multi-hop QA', 'Commonsense reasoning', 'Logical reasoning', 'Spatial and temporal reasoning', 'Symbolic and algorithmic', 'Math']

def get_identifier_prompting(dataset):
    '''
    Define the identifier as the experiment configuration setup, and assign values such as the prompting method, evaluation metric, and others.
    '''

    dataset_names = dataset['dataset']
    model_names = dataset['model_name']
    subset = dataset['subset']
    number_of_shots = dataset['number_of_shots']
    metric = dataset['metric']
    metric_value = dataset['metric_value']
    prompting_method = dataset['prompting_method']
    source_arxiv_id = dataset['table_source_arxiv_id']
    dataset_descriptions = dataset['dataset_description']
    table_index = dataset['table_index']
    categorization = dataset['categorization']

    identifiers = []
    for mn, sa, dn, s, ns, m, ti in zip(model_names, source_arxiv_id, dataset_names, subset, number_of_shots, metric, table_index):
        identifier = "_".join([mn, sa, dn, s, ns, m, str(ti)])
        identifiers.append(identifier)
    
    identifier_to_prompting_methods = defaultdict(list)
    for i, identifier in enumerate(identifiers):
        identifier_to_prompting_methods[identifier].append((prompting_method[i], metric_value[i], metric[i], model_names[i],
                                                            source_arxiv_id[i], dataset_names[i], subset[i], dataset_descriptions[i], i))

    for k in identifier_to_prompting_methods.keys():
        identifier_to_prompting_methods[k].append(categorization[identifier_to_prompting_methods[k][0][-1]])

    filtered_identifier_to_prompting_methods = {k: v for k, v in identifier_to_prompting_methods.items() if len(v) > 1}

    refiltered_identifier_to_prompting_methods = defaultdict(list) # pre-filter
    for k in filtered_identifier_to_prompting_methods.keys():
        to_include = False
        for t in filtered_identifier_to_prompting_methods[k]:
            if 'cot' in t[0].lower(): # HACK: rule-based
                if 'no' not in t[0].lower():
                    to_include = True
            elif 'chain' in t[0].lower() and 'thought' in t[0].lower():
                to_include = True
        if to_include:
            refiltered_identifier_to_prompting_methods[k] = filtered_identifier_to_prompting_methods[k]

    filtered_identifier_to_prompting_methods = refiltered_identifier_to_prompting_methods

    return filtered_identifier_to_prompting_methods


def get_prompt_annotations(args):
    cot_prompt_filter_list = []
    with open(args.cot_prompt_filtering_list_path, "r") as f:
        for i in f:
            cot_prompt_filter_list.append(i.strip().lower())            

    standard_prompt_filter_list = []
    with open(args.standard_prompt_filtering_list_path, "r") as f:
        for i in f:
            standard_prompt_filter_list.append(i.strip().lower())
    
    return cot_prompt_filter_list, standard_prompt_filter_list


def main(args):    
    dataset = load_dataset(args.hf_ds_path)
    dataset = process_ds(dataset, args.arxiv_id_to_venue, args.venue_filter)['train']

    identifier_to_prompting = get_identifier_prompting(dataset) 
    cot_prompt_filter_list, standard_prompt_filter_list = get_prompt_annotations(args)

    if args.classification_results_sprague:
        with open(args.classification_results_sprague, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = [line.strip().split("\t") for line in lines]
        for line in lines:
            k = line[0]
            v = line[1]
            if k in identifier_to_prompting and len(identifier_to_prompting[k]) > 0:
                identifier_to_prompting[k].pop()  # Remove the last element
            identifier_to_prompting[k].append(v)

    total_deltas_per_category = {}
    total_deltas_aggregated_by_paper_per_category = {}
    
    category_list = CATEGORIES_SPRAGUE if args.classification_results_sprague else CATEGORIES

    for category in category_list:
        for k in identifier_to_prompting.keys():
            if category in identifier_to_prompting[k][-1]: 
                prompting_methods_and_values = identifier_to_prompting[k][:-1]
                
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

                    if category not in total_deltas_per_category:
                        total_deltas_per_category[category] = []

                    if category not in total_deltas_aggregated_by_paper_per_category:
                        total_deltas_aggregated_by_paper_per_category[category] = {}
                    paper_key = prompting_methods_and_values[0][4]
                    if paper_key not in total_deltas_aggregated_by_paper_per_category[category]:
                        total_deltas_aggregated_by_paper_per_category[category][paper_key] = []
                    total_deltas_aggregated_by_paper_per_category[category][paper_key].append(delta)                 
                    total_deltas_per_category[category].append(delta)
                    
    for k in total_deltas_aggregated_by_paper_per_category.keys():
        for paper in total_deltas_aggregated_by_paper_per_category[k].keys():
            total_deltas_aggregated_by_paper_per_category[k][paper] = np.mean(total_deltas_aggregated_by_paper_per_category[k][paper])
    
    total_deltas = []
    for category in total_deltas_per_category.keys():
        total_deltas.extend(total_deltas_per_category[category])
    total_median = np.median(total_deltas)

    if os.path.exists(args.output_path) == False:
        os.makedirs(args.output_path)

    if args.classification_results_sprague:
        plot_output_path = os.path.join(args.output_path, 'to_cot_or_not_to_cot_verification_sprague_category.png')
        # desired_order = ['Entailment', 'Text classification', 'Generation', 'Mixed datasets', 'Encyclopedic knowledge', 'Context-aware QA', 'Multi-hop QA', 'Commonsense reasoning', 'Logical reasoning', 'Spatial and temporal reasoning', 'Symbolic and algorithmic', 'Math']
        desired_order = ['Commonsense reasoning', 'Logical reasoning', 'Spatial and temporal reasoning', 'Symbolic and algorithmic', 'Math'] # for truncated version
    else:
        plot_output_path = os.path.join(args.output_path, 'to_cot_or_not_to_cot_verification_our_category.png')
        desired_order = ['Coding', 'Instruction Following', 'Knowledge', 'Multilinguality', 'Multimodality', 'Math', 'Reasoning', 'Safety']

    create_boxplots(total_deltas_per_category, total_deltas_aggregated_by_paper_per_category, 
                    total_median, desired_order, plot_output_path)

    if args.classification_results_sprague:
        cot_total_deltas_output_path = os.path.join(args.output_path, 'cot_total_deltas_per_category_sprague_categories.json')
        cot_total_deltas_aggregated_by_paper_output_path = os.path.join(args.output_path, 'cot_total_deltas_aggregated_by_paper_per_category_sprague_categories.json')    
    else:
        cot_total_deltas_output_path = os.path.join(args.output_path, 'cot_total_deltas_per_category_our_categories.json')
        cot_total_deltas_aggregated_by_paper_output_path = os.path.join(args.output_path, 'cot_total_deltas_aggregated_by_paper_per_category_our_categories.json')

    with open(cot_total_deltas_output_path, "w") as f:
        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in total_deltas_per_category.items()}, f)
    with open(cot_total_deltas_aggregated_by_paper_output_path, "w") as f:
        json.dump({k: dict(v) for k, v in total_deltas_aggregated_by_paper_per_category.items()}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_ds_path', type=str, default='jungsoopark/LLMs-Performance-Data')
    parser.add_argument('--arxiv_id_to_venue', type=str, default='./metadata/arxiv_id_to_venue.json')
    parser.add_argument('--venue_filter', action='store_true', default=False)
    parser.add_argument('--classification_results_sprague', type=str, default=None)
    parser.add_argument('--cot_prompt_filtering_list_path', type=str, default='./classification_results/cot_prompt_filtering_list.txt')
    parser.add_argument('--standard_prompt_filtering_list_path', type=str, default='./classification_results/standard_prompt_filtering_list.txt')
    parser.add_argument('--output_path', type=str, default='./analysis_outputs/')
    args = parser.parse_args()
    main(args)
