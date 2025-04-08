import argparse
import json
import os

import ray

DOMAINS = ["cs.AI", "cs.CL", "cs.LG", "cs.CV"]

def filter_arxiv_metadata(paper_id: str, lookup_dict: dict):
    
    try:
        categories = lookup_dict[paper_id].split(' ')
    except Exception:
        return False
    
    to_save_domain = False    
    for category in categories:
        if category in DOMAINS:
            to_save_domain = True
            break

    return to_save_domain

@ray.remote
def process_chunk(chunk_input_directory: str, output_directory: str, lookup_dict: dict):
    for paper_path in os.listdir(chunk_input_directory):
        paper_id = paper_path.split("_")[0]
        to_save = filter_arxiv_metadata(paper_id, lookup_dict)
        
        if to_save:
            source_path = os.path.join(chunk_input_directory, paper_path)
            target_path = os.path.join(output_directory, paper_id)
            os.system(f'cp -r {source_path} {target_path}')

def main(extracted_root_path: str, output_directory: str, arxiv_metadata_path: str, prefix: str, num_cpus: int):

    chunk_folders = [os.path.join(extracted_root_path, folder) for folder in os.listdir(extracted_root_path)]
    if prefix is not None:
        chunk_folders = [folder for folder in chunk_folders if os.path.basename(folder).startswith(prefix)]
    
    chunk_folders.sort()

    metadata = []
    with open(arxiv_metadata_path, 'r') as f:
        for i in f:
            metadata.append(json.loads(i))

    lookup_dict = {}
    for i in metadata:
        lookup_dict[i['id']] = i['categories']

    ray.init(num_cpus=num_cpus)
    ray.get([process_chunk.remote(chunk_folder, output_directory, lookup_dict) for chunk_folder in chunk_folders])
    ray.shutdown()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_path', type=str)
    parser.add_argument('--extracted_ml_path', type=str)
    parser.add_argument('--arxiv_metadata_path', type=str)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--num_cpus', type=int, default=6)
    args = parser.parse_args()

    if not os.path.exists(args.extracted_ml_path):
        os.makedirs(args.extracted_ml_path)
    
    main(args.extracted_path, args.extracted_ml_path, args.arxiv_metadata_path, args.prefix, args.num_cpus)
