import argparse
import pandas as pd

from collections import defaultdict

from datasets import load_from_disk, Dataset


def remove_duplicates(data):

    original_length = len(data['table_source_arxiv_id'])
    total_row_strings = []
    for i in range(original_length):
        temp = ""
        for k in ['dataset_name', 'subset', 'number_of_shots', 'prompting_method', 'metric']:
            temp += data[k][i]
        total_row_strings.append(temp)

    assert len(total_row_strings) == original_length

    duplicate_indices_dict = defaultdict(list)
    for i, row in enumerate(total_row_strings):
        duplicate_indices_dict[row].append(i)

    duplicate_indices_dict = {k: v for k, v in duplicate_indices_dict.items() if len(v) > 1}

    duplicate_indices_lst = []
    for k in duplicate_indices_dict.keys():
        duplicate_indices_lst.extend(duplicate_indices_dict[k])

    filter_out_duplicate_idx = []
    for k in duplicate_indices_dict.keys():
        group_idx = duplicate_indices_dict[k]
        group_metrics = [float(data['metric_value'][j]) for j in group_idx]
        if len(set(group_metrics)) != 1: # remove the whole group
            filter_out_duplicate_idx.extend(group_idx)
        else:
            filter_out_duplicate_idx.extend(group_idx[1:]) # remove all but the first one

    filter_out_idx = list(set(filter_out_duplicate_idx))    

    for k in data.keys():
        data[k] = [v for i, v in enumerate(data[k]) if i not in filter_out_idx]

    return data


def main(args):
    dataset = load_from_disk(args.hf_ds_path)
    print("Original Length of the Dataset: ", len(dataset))

    data = {
        'table_source_arxiv_id': dataset['paper_id'],
        'table_source': dataset['table_source'],
        'dataset_name': dataset['dataset_name'],
        'original_extracted_dictionary': dataset['context_augmented_table_results_extracted'],
        'dataset_reference_arxiv_id': dataset['dataset_link'],
        'dataset_description_source': dataset['description_source'],
        'dataset_description': dataset['dataset_description'],
        'subset': [instance['context_augmented_table_results_extracted']['subset'] for instance in dataset],
        'model_name': [instance['context_augmented_table_results_extracted']['model_name'] for instance in dataset],
        'number_of_shots': [instance['context_augmented_table_results_extracted']['number_of_shots'] for instance in dataset],
        'prompting_method': [instance['context_augmented_table_results_extracted']['prompting_method'] for instance in dataset],
        'metric': dataset['standardized_metric'],
        'metric_value': dataset['adjusted_metric_value'],
    }

    data = remove_duplicates(data)
    print("Deduplicated Length of the Dataset: ", len(data['dataset_name']))

    filter_out_idx = []   
        
    for i, dataset_name in enumerate(data['dataset_name']):
        if dataset_name=='' or dataset_name is None or dataset_name == "<FAILED>" or dataset_name == "xx":
            filter_out_idx.append(i)
        
    for i, model_name in enumerate(data['model_name']):
        if model_name=='':
            filter_out_idx.append(i)
        
    for i, metric_name in enumerate(data['metric']):
        if metric_name=='' or metric_name is None or metric_name == "<FAILED>" or metric_name == "xx":
            filter_out_idx.append(i)
        
    filtered_data = {}
    for k in data.keys():
        filtered_data[k] = [v for i, v in enumerate(data[k]) if i not in filter_out_idx]

    for i, desc in enumerate(data['dataset_description']):
        if "<FAILED>" in desc:
            filter_out_idx.append(i)     

    for i, desc in enumerate(data['dataset_description']):
        if 'dataset summary' not in desc.lower() or 'task explanation' not in desc.lower():
            filter_out_idx.append(i)
    
    for i, prompting_method in enumerate(data['prompting_method']):
        if 'lora' in prompting_method.lower() or 'ft' in prompting_method.lower() or 'tuning' in prompting_method.lower():
            filter_out_idx.append(i)

    filtered_data = {}
    for k in data.keys():
        filtered_data[k] = [v for i, v in enumerate(data[k]) if i not in filter_out_idx]
    
    filtered_length = len(filtered_data['table_source_arxiv_id'])

    print("Filtered out Final Dataset Length: ", len(data['dataset_name']))

    for k in filtered_data.keys():
        assert len(filtered_data[k]) == filtered_length, f"Length of {k} is not equal to the filtered length of the dataset"        
    
    subsets = filtered_data['subset']

    for i, desc in enumerate(filtered_data['dataset_description']):
        if subsets[i] != 'xx':
            assert "Subset Description" in desc
    
    ds = Dataset.from_dict(filtered_data)
    ds.save_to_disk(args.hf_ds_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)
    args = parser.parse_args()
    main(args)
