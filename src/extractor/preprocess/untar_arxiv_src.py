import argparse
import os
import sys
import tarfile
import gzip

from glob import glob

import ray

def extract_tar_file(tar_path: str, extract_path: str, remove_original: bool) -> None:
    try:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=extract_path)
    except Exception as e:
        pass

    if remove_original:
        os.remove(tar_path)

def extract_gz_files(folder_path: str, remove_original: bool) -> None:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.gz') and not file.endswith('.pdf.gz'):
                gz_path = os.path.join(root, file)
                try:
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(gz_path[:-3], 'wb') as f_out:
                            f_out.write(f_in.read())
                except Exception as e:
                    pass

                if remove_original:
                    os.remove(gz_path)

@ray.remote
def process_tar_file(tar_file_path: str, extract_path: str, suffix: str) -> None:
    extract_path = os.path.join(extract_path, os.path.basename(tar_file_path).replace('.tar', ''))
    
    extract_tar_file(tar_file_path, extract_path, remove_original=False)
    extract_gz_files(extract_path, remove_original=True)

    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if not file.endswith('.pdf') and not file.endswith('.gz'):
                tar_path = os.path.join(root, file)
                tar_extract_path = tar_path + suffix
                extract_tar_file(tar_path, tar_extract_path, remove_original=True)
    
    pdf_files = glob(os.path.join(extract_path, '*.pdf'))
    for pdf_file in pdf_files:
        os.remove(pdf_file)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arXiv_download_path', type=str)
    parser.add_argument('--extracted_path', type=str)
    parser.add_argument('--num_cpus', type=int)
    args = parser.parse_args()

    tar_folder_path = args.arXiv_download_path
    extract_path = args.extracted_path
    num_cpus = args.num_cpus

    if not os.path.exists(tar_folder_path):
        print(f"Error: {tar_folder_path} does not exist.")
        sys.exit(1)

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    ray.init(num_cpus=num_cpus)
    
    tar_files = [file for file in os.listdir(tar_folder_path) if file.endswith('.tar')]
    tar_files_path = [os.path.join(tar_folder_path, file) for file in tar_files]
    
    results = ray.get([
        process_tar_file.remote(tar_file_path, extract_path, '_extracted')
        for tar_file_path in tar_files_path
    ])
    
    ray.shutdown()
