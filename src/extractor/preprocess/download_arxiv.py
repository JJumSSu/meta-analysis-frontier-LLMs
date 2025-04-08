import argparse
import os

import boto3
from botocore.exceptions import ClientError
import xml.etree.ElementTree as ET


class ArXivS3Downloader:
    def __init__(self, 
                 bucket_name='arxiv',
                 local_download_path='./',
                 region='us-east-1'):
                
        self.session = boto3.Session(region_name=region)
        self.s3_client = self.session.client('s3', 
            config=boto3.session.Config(
                s3={'use_sigv4': True},
                signature_version='s3v4'
            )
        )
        
        self.bucket_name = bucket_name
        self.local_download_path = local_download_path
        os.makedirs(self.local_download_path, exist_ok=True)

    def download_source_manifest(self):
        try:
            manifest_path = os.path.join(self.local_download_path, 'ArXiv_src_manifest.xml')

            self.s3_client.download_file(
                Bucket=self.bucket_name, 
                Key='src/arXiv_src_manifest.xml', 
                Filename=manifest_path,
                ExtraArgs={'RequestPayer': 'requester'}
            )
            
            tree = ET.parse(manifest_path)
            return tree.getroot()
        
        except ClientError as e:
            print(f"Error downloading manifest: {e}")
            print(f"Bucket: {self.bucket_name}")
            print(f"Region: {self.s3_client.meta.region_name}")
            
            if e.response['Error']['Code'] in ['403', 'InvalidAccessKeyId', 'SignatureDoesNotMatch']:
                print("Possible AWS credentials or permissions issue.")
                print("Please check:")
                print("1. AWS Access Key and Secret Key")
                print("2. IAM Permissions")
                print("3. Bucket access settings")
            
            return None

    def download_source_files(self, 
                               start_yymm='2023-01', 
                               end_yymm='2024-12'):

        
        start_yymm = start_yymm.replace('-', '')[-4:]
        end_yymm = end_yymm.replace('-', '')[-4:]
        
        manifest_root = self.download_source_manifest()
        if not manifest_root:
            print("Failed to download manifest. Check AWS configuration.")
            return
        
        total_downloaded = 0
        
        for file_elem in manifest_root.findall('file'):
            filename = file_elem.find('filename').text
            yymm = file_elem.find('yymm').text
            
            if not (start_yymm <= yymm <= end_yymm):
                continue
            
            local_filepath = os.path.join(
                self.local_download_path, 
                os.path.basename(filename)
            )
            
            try:
                self.s3_client.download_file(
                    Bucket=self.bucket_name, 
                    Key=filename, 
                    Filename=local_filepath,
                    ExtraArgs={'RequestPayer': 'requester'}
                )
                total_downloaded += 1
                
            except ClientError as e:
                print(f"Error downloading {filename}: {e}")
        
        print(f"Total files downloaded: {total_downloaded}")

def main():
    downloader = ArXivS3Downloader(
        bucket_name='arxiv',
        local_download_path=args.local_download_path,
        region=args.region
    )
    
    downloader.download_source_files(
        start_yymm=args.start_yymm, 
        end_yymm=args.end_yymm,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_download_path', type=str, required=True)
    parser.add_argument('--region', type=str, default='us-east-1')
    parser.add_argument('--start_yymm', type=str, default='2023-01')
    parser.add_argument('--end_yymm', type=str, default='2024-12')
    args = parser.parse_args()
    main()
