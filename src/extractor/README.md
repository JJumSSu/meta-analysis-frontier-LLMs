# Data Extraction from Arxiv Tex Sources

## Outline

This section explains how to use our pipeline for dataset extraction.
You can use a bash script to execute multiple components; however, this may take time and could result in unexpected errors.
We recommend testing the scripts with a shorter runtime (i.e. limited period) initially to ensure stability.
The following section also details how to run each component of the pipeline individually.

## Prerequisites

Before proceeding, make sure you have access to both the AWS API and the OpenAI API, as they are required for downloading and processing the data.

```bash
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export AZURE_OPENAI_KEY=your_azure_openai_api_key
```

You also need to download arXiv metadata snapsho (json file) for acquiring the metadata of downloaded arXiv sources. You can downaload the file from the following [url](https://www.kaggle.com/datasets/Cornell-University/arxiv).

## Running the Whole Pipeline

Coming Soon

## Running Each Component of the Pipeline Individually

The section below explains how to run each component individually in a sequential manner.
For illustration, we use arXiv sources from March 2023 and set "GPT-4" as the target model.

### Preprocessing

#### arXiv Sources (tex) Downloading

```bash

DOWNLOAD_PATH=YOUR_LOCAL_ORIGINAL_PATH
START_YYMM=2023-03
END_YYMM=2023-03

python -m extractor/preprocess/download_arxiv.py
    --local_download_path $DOWNLOAD_PATH
    --start_yymm $START_YYMM 
    --end_yymm $END_YYMM
```

#### Unzipping Downloaded Files

```bash
EXTRACTED_PATH=YOUR_LOCAL_EXTRACTED_PATH
NUM_CPUS=12

python -m extractor/preprocess/untar_arxiv_src.py
    --arXiv_download_path $DOWNLOAD_PATH
    --extracted_path $EXTRACTED_PATH 
    --num_cpus $NUM_CPUS
```

#### Filtering ML Domain Papers from Unzipped Files

```bash
FILTERED_PATH=YOUR_LOCAL_FILTERED_PATH
ARXIV_METADATA_PATH=DOWNLOADED_ARXIV_METADATA_PATH

python -m extractor/preprocess/filter_ml_domain.py
    --extracted_path $EXTRACTED_PATH
    --extracted_ml_path $FILTERED_PATH
    --arxiv_metadata_path $ARXIV_METADATA_PATH
    --num_cpus $NUM_CPUS
```

#### Extracting Tables from ML Papers

```bash
PREPROCESSING_DS_PATH=YOUR_PREPROCESSING_DS_PATH

python -m extractor/preprocess/extract_table.py
    --extracted_ml_path $FILTERED_PATH 
    --ml_table_ds $PREPROCESSING_DS_PATH
```

### Filtering

#### Filtering Leaderboard Tables

We use the `LLAMA3.1-70B-Instruct` model for filtering. This step can be time-consuming, so we highly recommend using the sharding dataset into multiple shards to parallelize the process across multiple GPUs. After processing, the results should be merged for further processing.

If you're extracting results for a single target model, it's more efficient to first filter the tables that contain its experimental results. We perform this step upfront to avoid repeating it for each target model individually.

In this example, we assume no sharding is used.

```bash
LEADERBOARD_DS=YOUR_LOCAL_LEADERBOARD_DS_PATH
CACHE_DIR=YOUR_CACHE_DIR
SHARD_IDX=0
TOTAL_SHARDS=1

python -m extractor/filter/filter_leaderboard_table.py
    --ml_table_ds $PREPROCESSING_DS_PATH
    --ml_leaderboard_table_ds $LEADERBOARD_DS
    --cache_dir $CACHE_DIR 
    --shard_idx $SHARD_IDX 
    --total_shards $TOTAL_SHARDS
```

#### Filtering Leaderboard Tables Containing Target Model's Records

This step filters the tables containing the target models. You can designate the target model you want for the extraction.
For each target model, please assign a corresponding keyword (e.g., `gpt` for GPT-4 and GPT-4o, `claude` for Claude 3, and `gemini` for Gemini 1.5 Pro) to streamline the filtering process.
We have included filtering prompts for each target model in the prompts folder. If you're using a new target model, you can add your custom filtering prompt there as well.

```bash
TARGET_MODEL_LEADERBOARD_DS=YOUR_LOCAL_TARGET_MODEL_LEADERBOARD_DS

python -m extractor/filter/filter_target_model_table.py
    --ml_leaderboard_table_ds $LEADERBOARD_DS
    --target_model_leaderboard_ds $TARGET_MODEL_LEADERBOARD_DS
    --model_keyword gpt 
    --model_name GPT-4 
    --prompt_path ./prompt/filter_target_model_gpt4.txt 
    --cache_dir $CACHE_DIR
```


### Extract

#### Schema-Driven Target Model Information Extraction

From this point forward, we use the OpenAI API (via Azure) to extract information from LaTeX sources.
You need to setup azure API configs.

This step identifies the experimental results of the target models and extracts information from table sources.

```bash

BACKEND=YOUR_AZURE_OPENAI_BACKEND_MODEL (e.g., gpt-4o)
DEPLOYMENT_NAME=YOUR_AZURE_OPENAI_DEPLOYMENT_NAME 
OPENAI_API_VERSION=YOUR_AZURE_OPENAI_API_VERSION
OPENAI_API_BASE=YOUR_AZURE_OPENAI_API_VERSION

TARGET_MODEL_EXTRACTED_DS=YOUR_LOCAL_TARGET_MODEL_EXTRACTED_DS

python -m extractor/extract/schema_extract_target_model 
    --hf_ds_path $TARGET_MODEL_LEADERBOARD_DS
    --hf_ds_output_path $TARGET_MODEL_EXTRACTED_DS
    --backend $BACKEND
    --deployment_name $DEPLOYMENT_NAME
    --openai_api_version $OPENAI_API_VERSION
    --openai_api_base $OPENAI_API_BASE
```

#### Context Augmentation

This stage enhances the data by integrating the complete contents of the paper with the records extracted in earlier steps. 
Following this, each cell from the tables is flattened, having been previously merged according to the model.

```bash

BACKEND=YOUR_AZURE_OPENAI_BACKEND_MODEL (e.g., gpt-4o)
DEPLOYMENT_NAME=YOUR_AZURE_OPENAI_DEPLOYMENT_NAME 
OPENAI_API_VERSION=YOUR_AZURE_OPENAI_API_VERSION
OPENAI_API_BASE=YOUR_AZURE_OPENAI_API_VERSION

TARGET_MODEL_EXTRACTED_AUGMENTED_DS=YOUR_LOCAL_TARGET_MODEL_EXTRACTED_AUGMENTED_DS

python3 -m extractor/extract/context_augment 
    --hf_ds_path $TARGET_MODEL_EXTRACTED_DS
    --hf_ds_output_path $TARGET_MODEL_EXTRACTED_AUGMENTED_DS
    --backend $BACKEND
    --deployment_name $DEPLOYMENT_NAME
    --openai_api_version $OPENAI_API_VERSION
    --openai_api_base $OPENAI_API_BASE
```


#### Metric Filtering and Adjustment

Using the extracted instances, we either discard invalid metrics or adjust their values as necessary.

```bash

TARGET_MODEL_EXTRACTED_AUGMENTED_FILTERED_DS=YOUR_LOCAL_TARGET_MODEL_EXTRACTED_AUGMENTED_FILTERED_DS

python3 -m extractor/extract/filter_and_adjust_metrics 
    --hf_ds_path $TARGET_MODEL_EXTRACTED_AUGMENTED_DS
    --hf_ds_output_path $TARGET_MODEL_EXTRACTED_AUGMENTED_FILTERED_DS
```


### Generate Description

In this step, we create descriptions of the dataset using either the LLM's knowledge or relevant references. If you have multiple target models, it is advisable to merge the extracted datasets from previous steps beforehand. This prevents redundant API calls for the same datasets (e.g., GPT-4 on MMLU, Gemini on MMLU).

#### Get Cited Dataset References

Initially, we obtain any referenced dataset citations from the extracted results for the given dataset. You should point out the downloaded arXiv sources directory to access the bib file.

```bash

TARGET_MODEL_EXTRACTED_BIBTEX_DS=YOUR_LOCAL_TARGET_MODEL_EXTRACTED_BIBTEX_DS
ARXIV_CITED_DOWNLOAD_PATH=YOUR_ARXIV_CITED_DOWNLOAD_PATH
DOWNLOADED_ARXIV_SOURCE_PATH=YOUR_DOWNLOADED_ARXIV_SOURCE_PATH # This path should correspond to the one you downloaded and unzipped during the preprocessing stage.

python3 -m extractor/generate_description/get_bibtex_paper 
    --hf_ds_path $TARGET_MODEL_EXTRACTED_AUGMENTED_FILTERED_DS
    --hf_ds_output_path $TARGET_MODEL_EXTRACTED_BIBTEX_DS
    --source_path $DOWNLOADED_ARXIV_SOURCE_PATH
    --source_save_path $ARXIV_CITED_DOWNLOAD_PATH
```


#### Canonicalizing Dataset Names (Optional)

We standardize the dataset names through character-based rule matching. Although this step is optional, it can help minimize duplicate dataset names with varying surface forms, thereby reducing API calls.


```bash

TARGET_MODEL_EXTRACTED_BIBTEX_CANONICALIZED_DS=YOUR_LOCAL_TARGET_MODEL_EXTRACTED_BIBTEX_CANONICALIZED_DS

python3 -m extractor/generate_description/canonicalize 
    --hf_ds_path $TARGET_MODEL_EXTRACTED_BIBTEX_DS
    --hf_ds_output_path $TARGET_MODEL_EXTRACTED_BIBTEX_CANONICALIZED_DS
```

#### Generate Description using LLM

Initially, we create descriptions using the LLM. If the model is unable to generate these due to a lack of knowledge, we move the failed attempt to the extraction stage. 

```bash

TARGET_MODEL_EXTRACTED_DESCRIPTION_GENERATED_DS=YOUR_LOCAL_TARGET_MODEL_EXTRACTED_DESCRIPTION_GENERATED

python3 -m extractor/generate_description/generate_description 
    --hf_ds_path $TARGET_MODEL_EXTRACTED_BIBTEX_CANONICALIZED_DS
    --hf_ds_output_path $TARGET_MODEL_EXTRACTED_DESCRIPTION_GENERATED_DS
```


#### Extract Description from References using LLM

Here, the model refers to the original paper or the cited dataset reference paper for guidance.

```bash

TARGET_MODEL_EXTRACTED_DESCRIPTION_EXTRACTED_DS=YOUR_LOCAL_TARGET_MODEL_EXTRACTED_DESCRIPTION_EXTRACTED

python3 -m extractor/generate_description/extract_description 
    --hf_ds_path $TARGET_MODEL_EXTRACTED_DESCRIPTION_GENERATED_DS
    --hf_ds_output_path $TARGET_MODEL_EXTRACTED_DESCRIPTION_EXTRACTED_DS
    --orig_source_path $DOWNLOADED_ARXIV_SOURCE_PATH
    --citation_source_path $ARXIV_CITED_DOWNLOAD_PATH
```

#### Final Post-Processing

Finally, we post-process the dataset. The post-processed file is the final output of our pipeline.

```bash
TARGET_MODEL_FINAL_EXTRACTED_DS=YOUR_TARGET_MODEL_FINAL_EXTRACTED_DS

python3 -m extractor/generate_description/post_process
    --hf_ds_path $TARGET_MODEL_EXTRACTED_DESCRIPTION_EXTRACTED_DS
    --hf_ds_output_path $TARGET_MODEL_FINAL_EXTRACTED_DS
```