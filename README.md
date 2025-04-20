# Malicious URL ML Pipeline

This repository implements an automated ML pipeline for analyzing malicious URLs. The code is refactored for ease of use, modularity, and automation. The project includes:

- **Automated Data Preprocessing:**  
  Loads, validates, and preprocesses the malicious URL dataset.
- **URL Feature Extraction:**  
  Modularized functions to extract enhanced features from each URL.
- **Multi-Model Training & Evaluation:**  
  Trains and evaluates several machine learning algorithms (LDA, Logistic Regression, SVM, Random Forest) concurrently.
- **Automated Visualization & Reporting:**  
  Creates and saves plots for feature correlations, model confusion matrices, and more.
- **Energy & CO₂ Tracking:**  
  Optionally tracks and reports energy and CO₂ emissions using CodeCarbon.

## Prerequisites

Make sure you have [Conda](https://docs.conda.io/en/latest/) installed before proceeding. The following packages are required:

- Python 3.8 (or later)
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib (for HPC only)
- codecarbon (must pip install)

## Setup Instructions for Local Computing

### 1. Clone the Repository

Clone this repository to your local machine:

```
git clone https://github.com/dfromond3/Malicious-URL-HPC.git
cd Malicious-URL-HPC
```

### 2. Create and Activate a Conda Environment

Create a new Conda environment with the desired Python version:

```
conda create -n ml_env python=3.10
conda activate ml_env
```

### 3. Install Required Dependencies

Using Conda for most packages:
```
conda install numpy scipy scikit-learn matplotlib pandas seaborn -c conda-forge
```

Using pip for CodeCarbon:
```
pip install codecarbon
```

Download the [Kaggle Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) and save it as _malicious_phish.csv_ in the same folder as the Python script.

### 4. Running the Script

To run the script, execute:
```
python intensive_ml_model.py
```
###

## Setup Instructions for HPC Computing

### 1. Clone the Repository

Clone this repository to your desired HPC directory:

```bash
git clone https://github.com/dfromond3/Malicious-URL-HPC.git
cd Malicious-URL-HPC
```

### 2. Create and Activate a Conda Environment

Create a new Conda environment with the desired Python version using a symlink (for example):

```
mv ~/.conda /storage/ice1/1/7/dfromond3/.conda
ln -s /storage/ice1/1/7/dfromond3/.conda ~/.conda

module load anaconda3
conda create --name hpc_env python=3.8 -y
conda activate hpc_env
```

### 3. Install Required Dependencies

Using Conda for most packages:
```
conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn joblib -y
```

Using pip for CodeCarbon:
```
pip install codecarbon
```

Download the [Kaggle Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) and save it as _malicious_phish.csv_ in the same folder as the Python script.

### 4. Running the Script

First ensure a job has started. To start a job, please execute one of the following (for example):

```
salloc -N1 --ntasks-per-node=4 -t1:00:00
salloc -N2 --ntasks-per-node=4 -t1:00:00
salloc --gres=gpu:H100:1 --ntasks-per-node=3
salloc --gres=gpu:H100:2 --ntasks-per-node=1
```
For documentation on how to start a job, please refer to [this webpage](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096).

To run the script after the job has started:
```
srun python intensive_ml_model_hpc.py
```
