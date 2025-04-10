import argparse
import numpy as np

from datasets import load_dataset
from collections import defaultdict

from utils import process_ds, create_boxplots


CATEGORIES = ['Tool Use', 'Multimodality', 'Math', 'Coding', 'Instruction Following', 'Safety', 'Knowledge', 'Reasoning', 'Multilinguality']


def get_identifier_to_prompting_methods(dataset):
    '''
    Define the identifier as the experiment configuration setup, and assign values such as the number of shots, evaluation metric, and others.
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
    for mn, sa, dn, s, pm, m, ti in zip(model_names, source_arxiv_id, dataset_names, subset, prompting_method, metric, table_index):
        identifier = "_".join([mn, sa, dn, s, pm, m, str(ti)])
        identifiers.append(identifier)

    identifier_to_prompting_methods = defaultdict(list)
    for i, identifier in enumerate(identifiers):
        identifier_to_prompting_methods[identifier].append((prompting_method[i], metric_value[i], metric[i], model_names[i],
                                                            source_arxiv_id[i], dataset_names[i], subset[i], dataset_descriptions[i], number_of_shots[i], i))
    
    for k in identifier_to_prompting_methods.keys():
        identifier_to_prompting_methods[k].append(categorization[identifier_to_prompting_methods[k][0][-1]])

    filtered_identifier_to_prompting_methods = {k: v for k, v in identifier_to_prompting_methods.items() if len(v) > 2}

    return filtered_identifier_to_prompting_methods


def main(args):
    dataset = load_dataset(args.hf_ds_path)['train']
    dataset = process_ds(dataset, args.arxiv_id_to_venue, args.venue_filter, process_few_shot=True)
    
    identifier_to_prompting = get_identifier_to_prompting_methods(dataset)

    total_deltas_for_less_shot_vs_more_shot_dict = {}
    total_deltas_for_zero_shot_vs_few_shot_dict = {}
    total_deltas_for_less_shot_vs_more_shot_aggregated_by_paper_per_category = {}
    total_deltas_for_zero_shot_vs_few_shot_aggregated_by_paper_per_category = {}

    for category in CATEGORIES:
        total_deltas_for_zero_shot_vs_few_shot = []
        total_deltas_for_less_shot_vs_more_shot = []

        for k in identifier_to_prompting.keys():
            if category in identifier_to_prompting[k][-1]: 
                prompting_methods_and_values = identifier_to_prompting[k][:-1] # get only the values
                prompting_methods_and_values.sort(key=lambda x: x[8])
                prompting_methods_and_values = [x for x in prompting_methods_and_values if x[8] < 100]     

                is_pass = False
                for instance in prompting_methods_and_values:
                    if str(instance[8]) == '0':
                        is_pass = True
                
                # zero-shot versus few-shot

                if is_pass:
                    do_not_append = False
                    zero_shot_performance = [instance[1] for instance in prompting_methods_and_values if str(instance[8]) == '0']
                    if len(zero_shot_performance) != 1: # two conflicting results from different papers
                        do_not_append = True
                    else:
                        zero_shot_performance = zero_shot_performance[0]
                        few_shot_performance = []
                        for instance in prompting_methods_and_values:
                            if str(instance[8]) == '0':
                                continue
                            else:
                                assert int(instance[8]) != 0
                                assert int(instance[8]) > 0
                                few_shot_performance.append(instance[1])
                        
                        if len(few_shot_performance) == 0:
                            continue

                        few_shot_performance = sum(few_shot_performance) / len(few_shot_performance)
                        delta = few_shot_performance - zero_shot_performance

                        if not do_not_append:
                            total_deltas_for_zero_shot_vs_few_shot.append(delta)
                            
                            if category not in total_deltas_for_zero_shot_vs_few_shot_aggregated_by_paper_per_category:
                                total_deltas_for_zero_shot_vs_few_shot_aggregated_by_paper_per_category[category] = {}
                            paper_key = prompting_methods_and_values[0][4]
                            if paper_key not in total_deltas_for_zero_shot_vs_few_shot_aggregated_by_paper_per_category[category]:
                                total_deltas_for_zero_shot_vs_few_shot_aggregated_by_paper_per_category[category][paper_key] = []
                            total_deltas_for_zero_shot_vs_few_shot_aggregated_by_paper_per_category[category][paper_key].append(delta)

                # less-shot versus more-shot

                deltas = []
                for i in range(len(prompting_methods_and_values)):
                    for j in range(i+1, len(prompting_methods_and_values)):
                        if prompting_methods_and_values[j][8] == prompting_methods_and_values[i][8]:
                            continue
                        cur_delta = prompting_methods_and_values[j][1] - prompting_methods_and_values[i][1]
                        assert int(prompting_methods_and_values[j][8]) - int(prompting_methods_and_values[i][8]) != 0
                        assert int(prompting_methods_and_values[j][8]) - int(prompting_methods_and_values[i][8]) > 0
                        deltas.append(cur_delta)

                if len(deltas) > 0:
                    delta = sum(deltas) / len(deltas)
                    total_deltas_for_less_shot_vs_more_shot.append(delta)

                    if category not in total_deltas_for_less_shot_vs_more_shot_aggregated_by_paper_per_category:
                        total_deltas_for_less_shot_vs_more_shot_aggregated_by_paper_per_category[category] = {}
                    paper_key = prompting_methods_and_values[0][4]
                    if paper_key not in total_deltas_for_less_shot_vs_more_shot_aggregated_by_paper_per_category[category]:
                        total_deltas_for_less_shot_vs_more_shot_aggregated_by_paper_per_category[category][paper_key] = []
                    total_deltas_for_less_shot_vs_more_shot_aggregated_by_paper_per_category[category][paper_key].append(delta)
                
        total_deltas_for_less_shot_vs_more_shot_dict[category] = total_deltas_for_less_shot_vs_more_shot
        total_deltas_for_zero_shot_vs_few_shot_dict[category] = total_deltas_for_zero_shot_vs_few_shot

    for k in total_deltas_for_zero_shot_vs_few_shot_aggregated_by_paper_per_category.keys():
        for paper in total_deltas_for_zero_shot_vs_few_shot_aggregated_by_paper_per_category[k].keys():
            total_deltas_for_zero_shot_vs_few_shot_aggregated_by_paper_per_category[k][paper] = np.mean(total_deltas_for_zero_shot_vs_few_shot_aggregated_by_paper_per_category[k][paper])

    for k in total_deltas_for_less_shot_vs_more_shot_aggregated_by_paper_per_category.keys():
        for paper in total_deltas_for_less_shot_vs_more_shot_aggregated_by_paper_per_category[k].keys():
            total_deltas_for_less_shot_vs_more_shot_aggregated_by_paper_per_category[k][paper] = np.mean(total_deltas_for_less_shot_vs_more_shot_aggregated_by_paper_per_category[k][paper])

    desired_order = ['Coding', 'Instruction Following', 'Knowledge', 'Multilinguality', 'Multimodality', 'Math', 'Reasoning', 'Safety']

    total_deltas = []
    for category in total_deltas_for_zero_shot_vs_few_shot_dict.keys():
        total_deltas.extend(total_deltas_for_zero_shot_vs_few_shot_dict[category])
    total_median_zero_shot_vs_few_shot = np.median(total_deltas)

    total_deltas = []
    for category in total_deltas_for_less_shot_vs_more_shot_dict.keys():
        total_deltas.extend(total_deltas_for_less_shot_vs_more_shot_dict[category])
    total_median_less_shot_vs_more_shot = np.median(total_deltas)

    output_path_zero_shot_vs_few_shot = f"{args.output_path}/icl_zero_shot_vs_few_shot.png"
    output_path_less_shot_vs_more_shot = f"{args.output_path}/icl_less_shot_vs_more_shot.png"

    create_boxplots(total_deltas_for_zero_shot_vs_few_shot_dict, 
                    total_deltas_for_zero_shot_vs_few_shot_aggregated_by_paper_per_category, 
                    total_median_zero_shot_vs_few_shot,
                    desired_order,
                    output_path_zero_shot_vs_few_shot,
                    prefix='ICL',
                    colors=['#B04B78', '#D28EAC']
                    )
    
    create_boxplots(total_deltas_for_less_shot_vs_more_shot_dict, 
                    total_deltas_for_less_shot_vs_more_shot_aggregated_by_paper_per_category, 
                    total_median_less_shot_vs_more_shot,
                    desired_order,
                    output_path_less_shot_vs_more_shot,
                    prefix='ICL',
                    colors=['#B04B78', '#D28EAC']
                    )                    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_ds_path', type=str, default='jungsoopark/LLMs-Performance-Data')
    parser.add_argument('--arxiv_id_to_venue', type=str, default='./metadata/arxiv_id_to_venue.json')
    parser.add_argument('--venue_filter', action='store_true', default=False)
    parser.add_argument('--output_path', type=str, default='./analysis_outputs/')
    args = parser.parse_args()
    main(args)
