# Drug Prescription Pattern Mining


## Overview

This repository contains the implementation of a data mining project focused on analyzing prescription patterns to uncover trends and associations in drug prescription data. By applying frequent pattern mining techniques, the project identifies co-prescription patterns, optimizes treatments, and highlights potential drug interactions. The analysis is based on a large dataset with over 239,000 prescription entries, providing actionable insights for healthcare professionals.

The project was developed as part of the Second Year Second Cycle in Artificial Intelligence and Data Science at the Higher School of Computer Science, Sidi Bel Abbes. It explores unsupervised learning approaches, specifically association rule learning, to enhance drug prescription insights in healthcare.

Key goals include:
- Preprocessing raw prescription data to remove noise and focus on relevant transactions.
- Implementing and comparing frequent pattern mining algorithms: ECLAT, PyECLAT, FP-Growth, and Apriori.
- Deploying user-friendly web interfaces for doctors and pharmacists using Streamlit to facilitate data-driven decision-making.

The full project report is available in [ML_Mini_Project.pdf](ML_Mini_Project.pdf).

## Features

- **Data Preprocessing**: Transaction filtering (2-10 items per transaction), duplicate removal, missing value handling, and feature engineering using techniques like MultiLabelBinarizer for binary matrix encoding.
- **Exploratory Data Analysis (EDA)**: Visualizations of physician specialties, years of practice, top prescribed drugs, and transaction distributions. Insights into drug frequency, co-occurrences, and prescription relationships.
- **Frequent Pattern Mining**: Implementation of ECLAT (vertical format with intersections), PyECLAT (optimized Python version with parallelization), FP-Growth (compressed FP-tree structure), and Apriori (candidate generation and pruning).
- **Model Comparison**: Evaluation based on support, confidence, and lift thresholds across sample and full datasets. ECLAT recommended for large-scale datasets due to its balance of performance and scalability.
- **Deployment**: Interactive web interfaces for doctors (patient info and prescription management) and pharmacists (medicine compatibility checks) built with Streamlit.
- **Scalability**: Designed for large datasets, with saved models for easy integration.

## Dataset

The dataset consists of prescription transactions, each containing a list of drugs prescribed to patients. Key features include:
- Physician specialties and years of practice.
- Drug names and prescription frequencies.
- Over 239,000 entries, preprocessed to focus on transactions with 2-10 drugs for efficiency.

Visualizations highlight distributions such as top 10 drugs, transactions with >10 drugs, and drugs in <50 transactions.

For privacy reasons, the raw dataset is not included in this repository. Refer to the project report for detailed descriptions and insights.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ADHAYA-Technos/drug-prescription-pattern-mining.git
   cd drug-prescription-pattern-mining
   ```

2. Install dependencies using pip (Python 3.8+ recommended):
   ```
   pip install -r requirements.txt
   ```

   Dependencies include:
   - `pandas`, `numpy` for data handling.
   - `mlxtend` for Apriori and FP-Growth implementations.
   - `pyECLAT` for optimized ECLAT.
   - `matplotlib`, `seaborn` for visualizations.
   - `streamlit` for web interfaces.
   - `scikit-learn` for preprocessing (e.g., MultiLabelBinarizer).
   - Database: SQLite or MySQL for record management.

## Usage

### Running the Analysis
- Execute preprocessing and EDA scripts:
  ```
  python src/preprocessing.py
  python src/eda.py
  ```
- Run frequent pattern mining:
  ```
  python src/mining.py --algorithm eclat --support 0.02 --confidence 0.14 --lift 2.2
  ```
  Supported algorithms: `eclat`, `pyeclat`, `fpgrowth`, `apriori`.

### Deploying the Web Interfaces
- Launch the Streamlit app:
  ```
  streamlit run app.py
  ```
- Access the interfaces at `http://localhost:8501`:
  - **Doctor Interface**: Enter patient details, search/add medicines with compatibility suggestions, and manage prescriptions.
  - **Pharmacy Interface**: Search medicines and view compatible options to ensure safe dispensing.

Saved models (e.g., frequent itemsets) are loaded for real-time suggestions.

## Algorithms and Approaches

- **ECLAT**: Uses vertical data format and intersections for efficient frequent itemset mining. Advantages: Memory-efficient for sparse data; Disadvantages: Computationally heavy for dense datasets.
- **PyECLAT**: Python-optimized ECLAT with parallelization and efficient data structures. Ideal for integration with data science workflows.
- **FP-Growth**: Builds an FP-tree to avoid candidate generation. Advantages: Scalable for large datasets; Disadvantages: Memory-intensive.
- **Apriori**: Iterative candidate generation with pruning. Advantages: Simple; Disadvantages: Inefficient for large data.

## Results

The models were evaluated on sample and full datasets using support, confidence, and lift thresholds. Below are summarized metrics:

### FP-Growth Metrics

| Algorithm          | Support Threshold | Confidence Threshold | Min Lift Threshold | Hyperparameters              |
|--------------------|-------------------|----------------------|--------------------|------------------------------|
| FPGrowth (Sample) | 0.01              | 0.10                 | 1.97               | min length: 2                |
| FPGrowth (Sample) | 0.10              | 0.32                 | 1.36               | min length: 2                |
| FPGrowth (Sample) | 0.15              | 0.46                 | 1.4                | min length: 2                |
| FPGrowth (Full)   | 0.02              | 0.12                 | 1.8                | min length: 3                |
| FPGrowth (Full)   | 0.12              | 0.35                 | 1.5                | min length: 3                |
| FPGrowth (Full)   | 0.18              | 0.50                 | 1.6                | min length: 3                |

### ECLAT Metrics

| Algorithm        | Support Threshold | Confidence Threshold | Min Lift Threshold | Hyperparameters              |
|------------------|-------------------|----------------------|--------------------|------------------------------|
| ECLAT (Sample)   | 0.01              | 0.12                 | 2.0                | min length: 2                |
| ECLAT (Sample)   | 0.08              | 0.25                 | 1.75               | min length: 2                |
| ECLAT (Sample)   | 0.13              | 0.35                 | 1.65               | min length: 2                |
| ECLAT (Full)     | 0.02              | 0.14                 | 2.2                | min length: 3                |
| ECLAT (Full)     | 0.09              | 0.30                 | 1.9                | min length: 3                |
| ECLAT (Full)     | 0.14              | 0.40                 | 1.8                | min length: 3                |

### PyECLAT Metrics

| Algorithm         | Support Threshold | Confidence Threshold | Min Lift Threshold | Hyperparameters              |
|-------------------|-------------------|----------------------|--------------------|------------------------------|
| PyECLAT (Sample)  | 0.01              | 0.10                 | 2.1                | min length: 2                |
| PyECLAT (Sample)  | 0.06              | 0.22                 | 1.95               | min length: 2                |
| PyECLAT (Sample)  | 0.11              | 0.34                 | 1.8                | min length: 2                |
| PyECLAT (Full)    | 0.03              | 0.15                 | 2.3                | min length: 3                |
| PyECLAT (Full)    | 0.10              | 0.28                 | 2.0                | min length: 3                |
| PyECLAT (Full)    | 0.16              | 0.38                 | 1.9                | min length: 3                |

### Apriori Metrics (Full Dataset)

| Feature Extraction | Support Threshold | Confidence Threshold | Min Lift Threshold | Hyperparameters              |
|--------------------|-------------------|----------------------|--------------------|------------------------------|
| Drugs              | 0.01              | 0.10                 | 1.97               | min length: 2                |
| Drugs              | 0.10              | 0.32                 | 1.36               | min length: 2                |
| Drugs              | 0.15              | 0.46                 | 1.4                | min length: 2                |

**Conclusion**: For the 230,000-row dataset, ECLAT is recommended for its optimal balance between performance, scalability, and quality of frequent itemsets.

## Technology Stack

- **Backend**: Python, Streamlit.
- **Libraries**: pandas, numpy, mlxtend, pyECLAT, matplotlib, scikit-learn.
- **Database**: SQLite/MySQL for patient and medicine records.
- **Deployment**: Local or cloud (e.g., Heroku, AWS).

## Contributors

- Yacine DAIT DEHANE
- Djaber BOUDAOUD
- Abdelillah SERGHINE
- Abdessalam OMARI

**Supervisor**: Dr. Nassima DIF

## Contact
For inquiries, contact ADHYA Tech at adhaya.technos@gmail.com
