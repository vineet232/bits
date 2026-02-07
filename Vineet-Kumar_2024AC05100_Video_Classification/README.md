# Assignment-1 Video Classification 
(Classical Machine Learning vs Deep Learning)

## Project Overview

This assignment project addresses the problem of human action recognition from videos using a customized subset of the UCF-101 dataset. 
The main objective is to design and evaluate a complete video classification pipeline and to compare two different paradigms:

1. Classical machine learning using handcrafted visual and motion features  
2. Deep learning using end-to-end trainable models  

The project covers dataset organization, preprocessing, feature extraction, model development, experimental evaluation, and a comparative analysis of both approaches.
## Project GitHub Repository

- **URL:** https://github.com/vineet232/bits

To download the project, clone the GitHub repository by running the following command in your terminal:

```bash
git clone https://github.com/vineet232/bits.git

```
This will also download the customized dataset consisting 3 classes:
1. PullUps
2. Punch
3. PushUps

## Project Directory Structure
```text

VINEET-KUMAR_2024AC05100_Video_Classification/
│
├── code/
│ ├── part_a_classical.ipynb
│ ├── part_a_classical.pdf
│ ├── part_b_deep_learning.ipynb
│ ├── part_b_deep_learning.pdf
│ ├── comparative_analysis.ipynb
│ ├── comparative_analysis.pdf
│ ├── feature_extraction.py
│ ├── data_loader.py
│ ├── models.py
│ ├── utils.py
│ └── requirements.txt
│
├── report/
│ └── Vineet-Kumar_2024AC05100_Comparative_Report.pdf
│
├── dataset_info/
│ ├── dataset/
│   ├── PullUps/
│   ├── Punch/
│   ├── PushUps/
│   └── splits/
│     ├── test.csv
│     ├── train.csv
│     └── val.csv
│
│ ├── dataset_url.txt
│ ├── dataset_description.md
│ ├── data_statistics.txt
│ └── sample_frames/
│
├── results/
│ ├── confusion_matrices/
│ ├── performance_plots/
│ ├── feature_visualizations/
│ ├── saved_models/
│ ├── stats_classical/
│ ├── stats_deep_learning/
│ └── saved_feature_matrices/
│
└── README.md
```

---

## Project Objectives

- To construct a classical video classification system using handcrafted features.  
- To implement deep learning models for automatic spatio-temporal representation learning.  
- To compare both paradigms under identical dataset conditions.  
- To analyze performance, computational cost, and practical trade-offs.  

---

## Dataset Description

The project uses a customized subset of the UCF-101 action recognition dataset consisting of three classes:

- PullUps  
- Punch  
- PushUps  

The dataset is organized inside `dataset_info/dataset/` with a clear folder structure for each action class.  
Predefined CSV files are provided in the `splits/` folder to define training, validation, and test sets.

Additional dataset-related documentation is available in:

- `dataset_url.txt` – source and reference  
- `dataset_description.md` – detailed description  
- `data_statistics.txt` – dataset summary  
- `sample_frames/` – example extracted frames  

---

## Methodology Summary

### Classical Approach (Part A)
- Video decoding and preprocessing  
- Uniform temporal sampling and spatial resizing  
- Handcrafted feature extraction (color, texture, shape, motion, temporal statistics)  
- Video-level feature aggregation  
- Training of classical classifiers such as SVM, Logistic Regression, k-NN, Random Forest, and Gradient Boosting  
- Quantitative evaluation and visualization  

### Deep Learning Approach (Part B)
- Frame sequence preparation  
- Design of neural architectures for spatial and temporal modeling  
- End-to-end training with validation monitoring  
- Performance evaluation and comparison with classical methods  

A separate notebook performs comparative analysis between both approaches.

---

## Results

All generated outputs are stored in the `results/` directory:

- `confusion_matrices/` – classification results and error patterns  
- `performance_plots/` – training curves and comparison graphs  
- `feature_visualizations/` – intermediate visual and motion representations  
- `saved_models/` – contains models saved after training.
- `stats_classical/` – consists classial models related saved parameters and values.
- `stats_deep_learning/` – consists deep learning models related saved parameters and values.
- `saved_feature_matrices/` – consists feature matrices created after feature extraction.
The findings are discussed in detail in the final report.

---

## Report

The complete project documentation, including background, methodology, experiments, results, and conclusions, is provided in:

`report/Vineet-Kumar_2024AC05100_Comparative_Report.pdf`

---

## Requirements

All software dependencies are listed in:

`code/requirements.txt`


This typically includes Python, OpenCV, NumPy, scikit-learn, and deep learning libraries.

---

## How to Use This Project

1. Install required packages using `requirements.txt` 
   Note: It is recommended to use Python virtual environment to avoid any conflict.
2. Run `part_a_classical.ipynb` for the classical pipeline  
3. Run `part_b_deep_learning.ipynb` for deep learning experiments  
4. Run `comparative_analysis.ipynb` for combined evaluation  
5. View generated outputs in the `results/` directory  

---

## Notes

- The assignment project follows a modular and well-organized structure to support clarity and reproducibility.  
- Dataset files, experimental code, results, and documentation are separated for clean project management.  
- All experiments use the same dataset splits to ensure fair comparison.

---

## Student Information

- Student Name: Vineet Kumar
- Roll No: 2024AC05100 
- Course: Video Analytics  
- Assignment: Video Classification






