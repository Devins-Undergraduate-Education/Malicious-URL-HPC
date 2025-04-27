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

## Setup Instructions for HPC Pipeline

### 1. Clone the Repository

Clone this repository to your local machine:

```
git clone -b UI_Dev https://github.com/Devins-Undergraduate-Education/Malicious-URL-HPC.git
cd Malicious-URL-HPC
```

### 2. Create and Activate a Conda Environment

Create a new Conda environment with the desired Python version:

```
conda create -n hpc_app python=3.8 -y
conda activate hpc_app
```

### 3. Install Required Dependencies

Using Conda for dependency installation:
```
conda install flask paramiko python-dotenv -c conda-forge -y
```
Download the [Kaggle Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) and save it as _malicious_phish.csv_ in an accessable folder.

### 4. Create a SECRET_KEY

Create a `.env` file with a random 32-byte token:
```
python -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))" > .env
```

### 5. Running the Script

To run the script, execute:
```
python app.py
```
> [!WARNING]  
> The script `app.py` assumes the user's `scratch` directory (where Conda is set-up) is `/storage/ice1/1/7/{USER}`. If this is not accurate, please update the `BASE_STORAGE` constant within `app.py`.

### 6. Run the User-Interface

- Navigate to the local host as displayed in the terminal output. (e.x. http://111.0.0.1:5001)
- Log-In to the UI with your Georgia Tech username/password
- Upload `malicious_phish.csv` to the User-Interface and select `Submit Job`

> [!NOTE]  
> To halt execution of the script and User-Interface, enter `CTRL+C` into the terminal.

> [!TIP]
> Depending on HPC configurations, model usage, or the dataset, more memory may be needed. To allocate more memory, increase the `SBATCH_MEMORY` variable in 2GB increments within the `app.py` file. 

### 7. Obtaining Results

- Navigate to the `Files` tab on [PACE Open On-Demand](https://ondemand-ice.pace.gatech.edu/)
- Within the `scratch` folder, navigate into the `Malicious-URL-HPC` folder

Plots and emissions created by `intensive_ml_model_hpc.py` will be available in the `plots` folder. 

Terminal outputs created by `intensive_ml_model_hpc.py` will be available in the `output-[DATE].out` files.

Any errors during the execution of the script(s) will be available in the `error-[DATE].err` files.

The script has finished execution when the `State` on the User-Interface for all `JobID`s are `COMPLETED`. Results will continue to populate into the above folders and files until the process has finished.
