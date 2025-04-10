# A Large-Scale, Evolving Literature Analysis of Frontier LLMs

🤗[Data](https://huggingface.co/datasets/jungsoopark/LLMEvalDB) |  📄[Paper](https://arxiv.org/abs/2502.18791) | 

This repository contains the code for (1) extracting experimental data related to target LLMs from arXiv sources (constructing LLMEvalDB), and (2) analyzing the extracted data, LLMEvalDB, as presented in our paper [Can LLMs Help Uncover Insights about LLMs? 
A Large-Scale, Evolving Literature Analysis of Frontier LLMs
](https://arxiv.org/abs/2502.18791).


## Updates

**[04/09/2025]** We have open-sourced our automated analysis pipeline based on our dataset.

**[04/08/2025]** We have open-sourced our pipeline for extracting data from arXiv sources.

**[02/26/2025]** We have uploaded our v1.0 LLMEvalDB to [huggingface](https://huggingface.co/datasets/jungsoopark/LLMEvalDB).

**[02/26/2025]** We have uploaded our preprint to [arXiv](https://arxiv.org/abs/2502.18791).

## Setup
```
git clone https://github.com/JJumSSu/meta-analysis-frontier-LLMs.git
cd meta-analysis-frontier-LLMs
pip install -r requirements.txt
```

## Information Extraction from arXiv Sources

To run the full pipeline or individual components, please refer to `READEME.md` from the `src/extractor` directory.

## Analysis

To run the analysis based on our dataset LLMEvalDB, please refer to `READEME.md` from the `src/analysis` directory.

## Citation

If you use our work and are inspired by our work, please cite our work:

```
@article{park2025seeing,
  title={Seeing the Forest for the Trees: A Large Scale, Continuously Updating Meta-Analysis of Frontier LLMs},
  author={Park, Jungsoo and Kang, Junmo and Stanovsky, Gabriel and Ritter, Alan},
  journal={arXiv preprint arXiv:2502.18791},
  year={2025}
}
```


