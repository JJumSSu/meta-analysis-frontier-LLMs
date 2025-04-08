import argparse
import os  
import re
import tarfile
import requests

from tqdm import tqdm

import arxiv

from datasets import load_from_disk
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer


def find_paper_title(citation_tag, bib_directory):  
    bib_lines = []
    for root, _, files in os.walk(bib_directory):  
        for file in files:  
            if file.endswith(".bib"):  
                bib_file_path = os.path.join(root, file)  
                with open(bib_file_path, 'r', encoding='utf-8') as bib_file:  
                    lines = bib_file.readlines()  
                    bib_lines.extend(lines)
    bib_lines = [line for line in bib_lines if not line.strip().startswith('%')]

    start_line_idx = -1
    for i, line in enumerate(bib_lines):  
        if line.strip().startswith(f"@") and citation_tag in line:  
            # Look for the title in the subsequent lines  
            if start_line_idx == -1:
                start_line_idx = i
            else:
                print("no title match")
                return None # invalid
    
    if start_line_idx == -1: # no tag match
        print("no title match")
        return None

    for j in range(start_line_idx, start_line_idx+10):  
        if 'title' in bib_lines[j].strip():  
            # Extract the title  
            title_line = bib_lines[j]
            match = re.search(r'title\s*=\s*{(.*)},?', title_line)
            if match:
                title = match.group(1).strip()
                return title
    
    print("no title match")
    return None

def search_title(title, f1_theshold=0.8):
    client = arxiv.Client()

    clean_title = title.replace('{', '').replace('}', '')

    # Search arxiv
    search = arxiv.Search(
        query=clean_title,
        max_results=1
    )

    results = client.results(search)

    try:
        paper = next(results)
        retrieved_title = paper.title

        def calculate_f1_score(title, clean_title):
            vectorizer = CountVectorizer().fit([title, clean_title])
            title_vector = vectorizer.transform([title]).toarray()
            clean_title_vector = vectorizer.transform([clean_title]).toarray()
            return f1_score(title_vector[0], clean_title_vector[0], average='macro')

        f1 = calculate_f1_score(retrieved_title, clean_title)
        if f1 > f1_theshold:
            arxiv_id = paper.entry_id.split('/')[-1]
            arxiv_id = arxiv_id[:10]
            return arxiv_id
        print("no title search found")
        return None
    except StopIteration:
        print("no title search found")
        return None

def download_arxiv_source(arxiv_id, 
                          save_dir):
    
    target_path = os.path.join(save_dir, arxiv_id)
    if os.path.exists(target_path):
        print('Already downloaded')
        return True
    
    # Construct the URL for the source files
    source_url = f'https://arxiv.org/e-print/{arxiv_id}'
    
    # Download the source files
    response = requests.get(source_url)
    
    if response.status_code == 200:
        # Save the tar file
        try:
            tar_path = os.path.join(save_dir, f'{arxiv_id}.tar.gz')
            with open(tar_path, 'wb') as f:
                f.write(response.content)
            
            # Extract the tar file
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=os.path.join(save_dir, arxiv_id))
            
            # Remove the tar file
            os.remove(tar_path)
            print(f"Successfully downloaded and extracted source files for {arxiv_id}")
            return True
        
        except Exception:
            print(f"Failed to extract downloaded {arxiv_id} source files")
            return False
    else:
        print(f"Failed to download {arxiv_id} source files. Status code: {response.status_code}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path')
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--source_save_path', type=str)
    
    args = parser.parse_args()
    
    ds = load_from_disk(args.hf_ds_path)

    print("Total instances: ", len(ds))
    
    dataset_links = []
    external_source_paths = []
    source_path = args.source_path

    for instance in tqdm(ds):

        dataset_citation_tag = instance['context_augmented_table_results_extracted']['dataset_citation_tag']

        if dataset_citation_tag == "xx":
            dataset_links.append('')
            external_source_paths.append('')
            continue

        table_paper_id = instance['paper_id']
        bib_directory = os.path.join(source_path, instance['paper_id'])

        try:
            title = find_paper_title(dataset_citation_tag, bib_directory)
        except UnicodeDecodeError:
            print('unicode error')
            title = None
        
        if title is None:
            dataset_links.append('')
            external_source_paths.append('')
            continue

        try:
            arxiv_id = search_title(title)
        except UnicodeDecodeError:
            print('unicode error')
            arxiv_id = None

        if arxiv_id is None:
            dataset_links.append('')
            external_source_paths.append('')
            continue

        is_success = download_arxiv_source(arxiv_id, args.source_save_path)

        if is_success:
            dataset_links.append(arxiv_id)
            external_source_paths.append(os.path.join(args.source_save_path, arxiv_id))
        else:
            dataset_links.append('')
            external_source_paths.append('')
            continue

    ds = ds.add_column('dataset_link', dataset_links)
    ds = ds.add_column('external_source_path', external_source_paths)
    ds.save_to_disk(args.hf_ds_output_path)
