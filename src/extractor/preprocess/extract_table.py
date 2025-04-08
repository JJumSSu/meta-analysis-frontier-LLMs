import argparse
import json
import logging
import os

from tqdm import tqdm

from datasets import Dataset

from tex.process_tex import extract, get_tables

logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
)

def extract_full_paper_and_table_src(paper_src_path: str):
    extracted_data = extract(paper_src_path)
    if extracted_data is not None:
        paper_id = paper_src_path.split('/')[-1].strip()
        extracted_data['paper_id'] = paper_id
    return extracted_data    


def main(ml_domain_path: str, output_path: str):
    logging.info(f"Extracting from arxiv src")
    domain_papers = os.listdir(ml_domain_path)
    domain_papers_dir = [os.path.join(ml_domain_path, paper_dir) for paper_dir in domain_papers]
    domain_papers_dir.sort()

    extracted_arxiv_srcs = []
    for paper_dir in tqdm(domain_papers_dir):
        result = extract_full_paper_and_table_src(paper_dir)
        if result is not None:
            extracted_arxiv_srcs.append(result)

    logging.info(f"Original length of papers: {len(domain_papers_dir)}")
    logging.info(f"Extracted length of papers from arxiv src: {len(extracted_arxiv_srcs)}")
    
    n = 0
    for i in extracted_arxiv_srcs:
        n += len(i['tables_list'])
    
    logging.info(f"Valid Table Latex from arxiv src: {n}")

    huggingface_dict = {}
    huggingface_dict['paper_id'] = [extracted_arxiv_src['paper_id'] for extracted_arxiv_src in extracted_arxiv_srcs]
    huggingface_dict['tables_list'] = [json.dumps(extracted_arxiv_src['tables_list']) for extracted_arxiv_src in extracted_arxiv_srcs]
    huggingface_dict['tables_index'] = [json.dumps(extracted_arxiv_src['tables_index']) for extracted_arxiv_src in extracted_arxiv_srcs]
    huggingface_dict['full_paper_latex_code'] = [extracted_arxiv_src['full_paper_latex_code'] for extracted_arxiv_src in extracted_arxiv_srcs]
    
    dataset = Dataset.from_dict(huggingface_dict)
    dataset.save_to_disk(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_ml_path', type=str)
    parser.add_argument('--ml_table_ds', type=str)
    args = parser.parse_args()

    main(args.extracted_ml_path, args.ml_table_ds)
    logging.info("Finished extracting tables from arxiv sources")
