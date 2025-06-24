# Intrusion Detection System Analysis Script (merged_ids_analysis.py)

## Description

This Python script provides a comprehensive framework for analyzing intrusion detection datasets. It supports both the Kaggle and UNSW-NB15 datasets and allows for various preprocessing steps, model training and evaluation, and feature selection techniques. The script outputs a detailed summary of the analysis results to a text file.

## Features

- Supports Kaggle and UNSW-NB15 datasets.
- Includes basic and advanced Exploratory Data Analysis (EDA).
- Compares model performance before and after IQR-based outlier removal.
- Compares Random Forest performance with and without feature scaling.
- Implements correlation-based feature selection.
- Implements dynamic SHAP-based top 10 feature selection.
- Trains and evaluates multiple models:
    - Base Random Forest
    - Tuned Random Forest (using GridSearchCV)
    - PCA + Base Random Forest
    - Artificial Neural Network (ANN) with SMOTE (optional, requires TensorFlow)
    - Fuzzy C-Means clustering (optional, requires scikit-fuzzy)
- Performs SHAP analysis on the best Tuned Random Forest model (optional, requires SHAP).
- Logs all results to a summary text file.
- Suppresses the saving of plot images by default.

## Datasets Used

- **UNSW-NB15:** Created by the Australian Centre for Cyber Security (ACCS).
  - Link: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Cybersecurity IDS Dataset (Kaggle):** Provided by user dnkumars on Kaggle.
  - Link: https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset/

## Environment Setup

This project uses a Conda environment defined in `tf_gpu_env.yml`. To set up the environment, you need to have Conda installed.

1.  Navigate to the directory containing the `tf_gpu_env.yml` file in your terminal.
2.  Create the Conda environment using the provided file:
    ```bash
    conda env create -f tf_gpu_env.yml
    ```
3.  Activate the newly created environment:
    ```bash
    conda activate tf_gpu_env
    ```

This will install all the necessary libraries, including TensorFlow (with GPU support if available and configured), scikit-learn, pandas, numpy, matplotlib, seaborn, scipy, scikit-fuzzy, and shap.

## Prerequisites

- Conda installed.
- Access to the `tf_gpu_env.yml` file.

## How to Run

Navigate to the directory containing the `merged_ids_analysis.py` script in your terminal.

The script requires specifying the dataset name, input directory, and output directory.

**Basic Usage:**

```bash
python merged_ids_analysis.py --dataset_name [kaggle or unsw] --input_dir /path/to/your/data --output_dir ./analysis_results
```

**Example for Kaggle Dataset:**

Assuming `cybersecurity_intrusion_data.csv` is in the current directory:
```bash
python merged_ids_analysis.py --dataset_name kaggle --input_dir . --output_dir ./kaggle_results
```

**Example for UNSW-NB15 Dataset:**

Assuming `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` are in the current directory:
```bash
python merged_ids_analysis.py --dataset_name unsw --input_dir . --output_dir ./unsw_results
```

**Using Optional Flags:**

You can enable additional analyses using flags:

- `--sample_size [int]`: Number of samples to use from the dataset. Use `0` for the full dataset. **Default: 20000**.
- `--run_ann`: Include ANN+SMOTE analysis. **Default: False**.
- `--run_fuzzy`: Include Fuzzy C-Means analysis. **Default: False**.
- `--run_shap`: Include SHAP analysis. **Default: False**.
- `--feature_selection_corr_threshold [float]`: Correlation threshold for feature selection. If > 0, features with absolute correlation above this threshold will be dropped. **Default: 0.0 (disabled)**.
- `--run_shap_selection`: Run an additional analysis using only the top 10 features dynamically identified by SHAP from the initial run on original data. **Default: False**.

**Example with multiple options:**

```bash
python merged_ids_analysis.py --dataset_name unsw --input_dir . --output_dir ./unsw_analysis --sample_size 20000 --run_ann --run_fuzzy --run_shap --feature_selection_corr_threshold 0.9 --run_shap_selection
```

**Help Message:**

To see all available arguments and their descriptions, use the help flag:
```bash
python merged_ids_analysis.py --help
```

## Output

The script creates the specified output directory (if it doesn't exist) and saves the following:

- `summary.txt`: A detailed text file containing all EDA results, model performance metrics (accuracy, classification reports, confusion matrices, ROC AUC), training/prediction times, model sizes, and summaries of feature selection steps for each analysis run (original data, after outlier removal, dynamic SHAP selection).
- Trained models (`.pkl` or `.keras`) for the best Tuned Random Forest and ANN models.
- Fuzzy C-Means cluster centers (`.pkl`).

Note: Plot images (confusion matrices, ROC curves, SHAP plots) are generated in memory but are **not saved to disk** by default.

## Analysis Stages Explained

The script performs analysis in several stages for comparison:

1.  **Original Data:** The full analysis pipeline is run on the dataset as loaded (after initial cleaning).
2.  **After Outlier Removal:** If outliers are detected and removed using the IQR method, the full analysis pipeline is run again on this reduced dataset. This allows comparison of model performance with and without outliers.
3.  **Dynamic SHAP Top 10 Features:** If `--run_shap_selection` is used, the script identifies the top 10 most important features from the SHAP analysis performed on the original data. It then runs **only the Tuned Random Forest model** (scaled and unscaled) using just this subset of features. This shows the trade-off between performance and using a minimal, highly relevant feature set.

## Feature Selection Explained

- **Correlation-Based:** Removes features that are highly correlated with other features (above the specified threshold), aiming to reduce multicollinearity and redundancy.
- **Dynamic SHAP-Based:** Selects features based on their impact on the model's predictions as determined by SHAP values. This aims to identify the most influential features for the model's decision-making.
