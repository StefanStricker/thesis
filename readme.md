# Evaluating Generalization of CNN Architectures under Domain Shift in Trash Classification

This repository contains the codebase, experiments, and analysis for for the bachlor's thesis. The thesis investigates how different Convolutional Neural Network architectures affect robustness and generalization performance under domain shift.

---

Architectures tested:
 - ResNet-50
 - DenseNet-121
 - ConvNeXt-tiny

---

## Project Folder Structure

```
├── data/                               # Folder for data storage and preprocessing
│   ├── datasets/                       # Folder for storing datasets
│   │    ├── trashnet                   # Preprocessed TrashNet dataset   
│   │    └── trashvariety               # Preprocessed self-collected dataset
│   ├── hf_cache/                       # Huggingface cache
│   ├── subgroups/                      # Subgroups Folder for evaluation  
│   ├── subgroups.csv                   # Subgroups file for folder creation
│   ├── TrashNet_pre/                   # TrashNet dataset preprocessed
│   ├── download_self_collected.py      # Download self-collected dataset
│   └── trashnet_prep.py                # Preprocess TrashNet dataset
├── demo/                               # Demo folder
│    └── app.py                         # Demo application script
├── images/                             # Images folder
├── notebooks/                          # Notebooks folder
│   ├── evaluation.ipynb                # Model Evaluation notebook creates CSV files
│   ├── exploration.ipynb               # Explores the results of the evaluation notebook
│   ├── model_train.iypnb               # Trains all models
│   └── vis.ipynb                       # Creates Visualization of data sugmentations
├── results/                            # Results folder
│   ├── class_metrics.csv               # Class metrics for individual random seed runs 
│   ├── class_metrics_agg.csv           # Aggregated class metrics
│   ├── model_metrics_agg.csv           # Aggregated model metrics
│   ├── model_metrics.csv               # Model metrics for individual random seed runs
│   ├── subgroup_metrics.csv            # Subgroup metrics for individual random seed runs
│   └── subgroup_metrics_agg.csv        # Aggregated subgroup metrics
├── trained_models/                     # Folder to store trained models
├── .gitignore                          # Gitignore file    
├── requirements.txt                    # Requirements file
```

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

    python notebooks/model.train.iypnb

### 2. Evaluate models 

    python notebooks/evaluation.ipynb

## Application demonstration

The web-based demonstration application showcases trained classification models in an interactive setting. The demo allows users to upload images and view the predicted class probabilities for the best performing model Resnet50 with photometric augmentations as well as the baseline model for comparison.

To run the demo on local machine, the model weights for resnet50_baseline_seed64.pth and resnet_photo_seed64.pth need to be saved in the `trained_models` folder.

### Run the demo

    streamlit run demo/app.py