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

