# A Large Scale, Continuously Updating Meta-Analysis of Frontier LLMs

🤗[Data](https://huggingface.co/datasets/jungsoopark/LLMs-Performance-Data) |  📄[Paper](https://arxiv.org/abs/2502.18791) | 

This repository contains the code for (1) extracting experimental data related to target LLMs from arXiv sources, and (2) analyzing the extracted data, as presented in our paper [Seeing the Forest for the Trees: A Large Scale, Continuously Updating Meta-Analysis of Frontier LLMs
](https://arxiv.org/abs/2502.18791).


## Updates

**[04/08/2025]** We have open-sourced our pipeline for extracting data from arXiv sources.

**[02/26/2025]** We have uploaded our v1.0 dataset to [huggingface](https://huggingface.co/datasets/jungsoopark/LLMs-Performance-Data).

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

Code will be available soon.


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


