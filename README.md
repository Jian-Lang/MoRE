# MoRE: Mixture of Retrieval-Augmented Multimodal Experts

This repo is the official implementation of *Biting Off More Than You Can Detect: Retrieval-Augmented Multimodal Experts for Short Video Hate Detection* accepted by WWW 2025.

## Source Code Structure

```bash
data        # dir of each dataset
- HateMM 
- MultiHateClip     # i.e., MHClip-B and MHClip-Y
    - en
    - zh

retrieval   # code of retrieval

src         # code of MoRE
- config    # training config
- model     # model implementation
- utils     # training utils
- data      # dataloader of MoRE
```

## Dataset

We provide video IDs for each dataset in both temporal and five-fold splits. Due to copyright restrictions, the raw datasets are not included. You can obtain the datasets from their respective original project sites.

### HateMM

Access the full dataset from [hate-alert/HateMM](https://github.com/hate-alert/HateMM).

### MHClip-B and MHClip-Y

Access the full dataset from [Social-AI-Studio/MultiHateClip: Official repository for ACM Multimedia'24 paper "MultiHateClip: A Multilingual Benchmark Dataset for Hateful Video Detection on YouTube and Bilibili"](https://github.com/social-ai-studio/multihateclip).

# Usage

## Requirements

To set up the environment, run the following commands:

```bash
conda create --name py312 python=3.12
pip install torch transformers tqdm loguru pandas torchmetrics scikit-learn colorama wandb hydra-core
```

## Data Preprocess

1. Sample 16 frames from each video in the dataset.

2. Extract on-screen text from keyframes using Paddle-OCR.

3. Extract audio transcripts from video audio using Whisper-v3.

4. Encode visual feature from each video using a pre-trained ViT model.

5. Encode audio feature to MFCC with libsora.

6. Encode textual feature using a pre-trained BERT model.


## Retrieval

1. Encode audio transcirpt using BERT to make audio memory bank.

2. Encode title and description using BERT to make textual memory bank.

3. Encode 16 frames using ViT to make visual memory bank.

```bash
# conduct retrieval
python retrieve/make_retrieval_result.py
```

## Run

```bash
# Run ExMRD for the HateMM dataset
python src/main.py --config-name HateMM_MoRE

# Run ExMRD for the MHClip-Y dataset
python src/main.py --config-name MHClipEN_MoRE

# Run ExMRD for the MHClip-B dataset
python src/main.py --config-name MHClipZH_MoRE
```
