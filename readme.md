# Evaluating Generalization of CNN Architectures under Domain Shift in Trash Classification

This repository contains the codebase, experiments, and analysis for for the bachlor's thesis. The thesis investigates how different Convolutional Neural Network architectures affect robustness and generalization performance under domain shift.

---

Architectures tested:
 - ResNet-50
 - DenseNet-121
 - ConvNeXt-tiny

---

## Installation guide

### 1. Clone Repository


    git clone https://github.com/StefanStricker/thesis.git
    cd thesis


### 2. Create Virtual Environments

    python3 -m venv/venv
    source venv/bin/activate

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Dataset setup

#### A. Split TrashNet dataset into train/test/val

    python data/trashnet_prep.py

The TrashNet resized dataset is already in the repo (source https://github.com/garythung/trashnet)    

#### B. Download target dataset from huggingface

    python data/download_self_collected.py

The self-collected target dataset is available on hugging face https://huggingface.co/datasets/StefanStricker/trashvariety (~2.05GB)

## How to recreate the experiments


### 1. Train models (this trains all 12 models tested)

    python model.train.iypnb

### 2. Evaluate models 

    python evaluation.ipynb