# CY 530 Assignment 2 — Encrypted Darknet Traffic Detection

**Author:** Caliandra Facio
**Course:** CY 530 
**Topic:** Machine-Learning Based Detection of Encrypted Malicious Traffic

## Overview

This repository contains the code and results for an analysis of five machine learning classifiers for binary detection of darknet traffic on the CIC-Darknet2020 dataset:

- Decision Tree (baseline replication)
- Random Forest (baseline replication)
- XGBoost (extension)
- LightGBM (extension)
- Multilayer Perceptron (extension)

All models are trained on the same encoded feature set, evaluated on the same stratified 80/20 train-test split, and compared across accuracy, precision, recall, F1 score, false positive rate (FPR), AUC, and training time.

**Expected runtime:** approximately 90 seconds on a modern laptop.

## Expected Output

The script produces `results_summary.csv` with the following table:

| Model          | Accuracy | Precision | Recall | F1     | FPR    | AUC    | TrainTime_s |
|----------------|----------|-----------|--------|--------|--------|--------|-------------|
| Decision Tree  | 0.9992   | 0.9973    | 0.9977 | 0.9975 | 0.0006 | 0.9986 | 12.3        |
| Random Forest  | 0.9990   | 0.9992    | 0.9949 | 0.9970 | 0.0002 | 1.0    | 39.1        |
| XGBoost        | 0.9996   | 0.9994    | 0.9981 | 0.9988 | 0.0001 | 1.0    | 4.0         |
| LightGBM       | 0.9996   | 0.9996    | 0.9979 | 0.9988 | 0.0001 | 1.0    | 4.3         |
| MLP            | 0.9986   | 0.9992    | 0.9928 | 0.9960 | 0.0002 | 0.9999 | 25.8        |

## Requirements

- Anaconda or Miniconda
- Python 3.10
- Approximately 5 GB free disk space

## Setup

### 1. Create the Conda environment

conda env create -f environment.yml
conda activate darknet-env

### 2. Download the dataset

The dataset is not included in this repository due to its size. Download `darknet_dataset_processed_encoded.csv` from one of the following sources:

- **UNB CIC:** https://www.unb.ca/cic/datasets/darknet2020.html
- **Baseline repository:** https://github.com/mateuscmarim/Darknet-traffic-classification

Place the file in the `dataset/` folder:

