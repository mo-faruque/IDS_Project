# -*- coding: utf-8 -*-
"""
Merged Intrusion Detection System Analysis Script

This script combines analyses from cs519_ids_project.py and cs519_ids_project1_unsw.py.
It allows selecting the dataset (kaggle or unsw) and specifying input/output directories
via command-line arguments.

Example Usage:
python merged_ids_analysis.py --dataset_name kaggle --input_dir . --output_dir ./kaggle_results
python merged_ids_analysis.py --dataset_name unsw --input_dir . --output_dir ./unsw_results
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import warnings
import traceback
import sys
from contextlib import redirect_stdout

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve # Added learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, silhouette_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE

# TensorFlow imports (optional, based on ANN usage)
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info/warning messages
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN custom operations
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. ANN model will be skipped.")

# Fuzzy C-Means imports
try:
    import skfuzzy as fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("Warning: scikit-fuzzy not found. Fuzzy C-Means clustering will be skipped.")

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not found. Explainability analysis will be skipped.")

# Other imports
from scipy.stats import mode
from pandas.plotting import scatter_matrix

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --- Helper Function to Save Plots ---
def save_plot(figure, filename, output_dir):
    """Saves the given matplotlib figure to the output directory."""
    filepath = os.path.join(output_dir, filename)
    try:
        # Ensure the figure object is used if available, otherwise use plt's current figure
        if figure is None:
             figure = plt.gcf() # Get current figure if None was passed
        figure.savefig(filepath, bbox_inches='tight', dpi=150) # Save the figure
        print(f"Plot saved to: {filepath}")
        plt.close(figure) # Close the figure after saving
    except Exception as e:
        print(f"Error saving/closing plot {filepath}: {e}")
        if figure:
             plt.close(figure) # Attempt to close even if saving failed

# --- Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Intrusion Detection System Analysis")
    parser.add_argument("--dataset_name", type=str, required=True, choices=['kaggle', 'unsw'],
                        help="Name of the dataset to use ('kaggle' or 'unsw').")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the input dataset CSV files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results (summary.txt and plots).")
    parser.add_argument("--sample_size", type=int, default=20000,
                        help="Number of samples to use from the dataset (default: 20000). Set to 0 for full dataset.")
    parser.add_argument("--run_ann", action='store_true',
                        help="Run ANN+SMOTE analysis (requires TensorFlow).")
    parser.add_argument("--run_fuzzy", action='store_true',
                        help="Run Fuzzy C-Means analysis (requires scikit-fuzzy).")
    parser.add_argument("--run_shap", action='store_true',
                        help="Run SHAP analysis (requires SHAP, can be time-consuming).")
    parser.add_argument("--feature_selection_corr_threshold", type=float, default=0.0,
                        help="Correlation threshold for feature selection (e.g., 0.9). If > 0, features with absolute correlation above this threshold will be dropped. Default 0.0 (disabled).")
    parser.add_argument("--run_shap_selection", action='store_true',
                        help="Run full analysis using only top 10 features identified by previous SHAP analysis on original data.")

    args = parser.parse_args()
    return args

# --- Data Loading ---
def load_data(dataset_name, input_dir, sample_size):
    """Loads and performs initial cleaning for the specified dataset."""
    print(f"\n--- Loading Dataset: {dataset_name.upper()} ---")
    df = None
    target_col = None
    cols_to_drop = []

    if dataset_name == 'kaggle':
        file_path = os.path.join(input_dir, "cybersecurity_intrusion_data.csv")
        target_col = 'attack_detected'
        # Define columns to drop for Kaggle dataset
        cols_to_drop = ['session_id'] # Add other potential high-cardinality/ID columns if needed
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f"Successfully loaded {file_path}")
        except FileNotFoundError:
            print(f"Error: Kaggle dataset file not found at {file_path}")
            return None, None
        except Exception as e:
            print(f"An error occurred while loading the Kaggle dataset: {e}")
            return None, None

    elif dataset_name == 'unsw':
        train_path = os.path.join(input_dir, "UNSW_NB15_training-set.csv")
        test_path = os.path.join(input_dir, "UNSW_NB15_testing-set.csv")
        target_col = 'label'
        # Define columns to drop for UNSW dataset
        cols_to_drop = ['id', 'attack_cat']
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            df = pd.concat([df_train, df_test], ignore_index=True)
            print(f"Successfully loaded and combined {train_path} and {test_path}")
        except FileNotFoundError:
            print(f"Error: UNSW dataset file(s) not found in {input_dir}")
            return None, None
        except Exception as e:
            print(f"An error occurred while loading the UNSW dataset: {e}")
            return None, None

    if df is None:
        return None, None

    print(f"Original dataset shape: {df.shape}")

    # --- Data Cleaning: Handle potential mixed types more robustly (especially for Kaggle) ---
    if dataset_name == 'kaggle':
        print("Attempting conversion of object columns to numeric for Kaggle dataset...")
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Try converting to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # If conversion resulted in all NaNs, maybe it wasn't numeric
                if df[col].isnull().all():
                   df_reloaded_col = pd.read_csv(file_path, usecols=[col], low_memory=False)[col]
                   df[col] = df_reloaded_col # Reload original object column
            except ValueError:
                pass # Keep as object if conversion fails
        print("Numeric conversion attempt complete.")

    # Drop specified columns
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        print(f"Dropping columns: {existing_cols_to_drop}")
        df = df.drop(columns=existing_cols_to_drop)
    else:
        print(f"Columns specified for dropping ({cols_to_drop}) not all found.")

    # Sample dataset
    if sample_size > 0 and df.shape[0] > sample_size:
        print(f"Sampling dataset to {sample_size} rows...")
        df = df.sample(n=sample_size, random_state=42).copy()
    elif sample_size > 0:
         print(f"Dataset has fewer than {sample_size} rows ({df.shape[0]}), using the full dataset.")
         df = df.copy() # Ensure it's a copy
    else:
        print("Using the full dataset (sample_size=0).")
        df = df.copy() # Ensure it's a copy

    print(f"Dataset shape after initial processing: {df.shape}")

    # Check if target column exists
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in the DataFrame after loading/cleaning.")
        return None, None

    # Basic check for target variable type (needs to be suitable for classification)
    if not pd.api.types.is_numeric_dtype(df[target_col]) and not pd.api.types.is_categorical_dtype(df[target_col]) and not pd.api.types.is_object_dtype(df[target_col]):
         print(f"Warning: Target column '{target_col}' has an unusual type: {df[target_col].dtype}. Ensure it's suitable for classification.")
    # Convert target to integer if it looks like binary numeric (0.0/1.0)
    if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() == 2:
        if set(df[target_col].unique()) == {0.0, 1.0}:
             print(f"Converting target column '{target_col}' from float to integer.")
             df[target_col] = df[target_col].astype(int)

    print("\nDataset Head:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print(f"\nTarget Column ('{target_col}') Distribution:")
    print(df[target_col].value_counts(normalize=True))

    return df, target_col


# --- EDA Functions ---
def perform_basic_eda(df, target_col):
    """Performs and prints basic EDA results."""
    print("\n--- Basic Exploratory Data Analysis (EDA) ---")
    if df is None:
        print("DataFrame not available for Basic EDA.")
        return

    print("\n=== Dataset Info ===")
    # Use StringIO to capture info output
    from io import StringIO
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    print(info_buffer.getvalue())

    print("\n=== Statistical Summary (Numeric Columns) ===")
    # Limit precision for cleaner output in summary.txt
    with pd.option_context('display.float_format', '{:,.2f}'.format):
        print(df.describe(include=np.number))

    print("\n=== Statistical Summary (Object Columns) ===")
    # Describe object columns if they exist
    if not df.select_dtypes(include=['object']).empty:
        print(df.describe(include=['object']))
    else:
        print("No object columns found.")

    print("\n=== Missing Values per Column (Showing > 0) ===")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print(missing_values)
    else:
        print("No missing values found.")

    if target_col and target_col in df.columns:
        print(f"\n=== Target Variable ('{target_col}') Distribution ===")
        print("Value Counts:")
        print(df[target_col].value_counts())
        print("\nNormalized:")
        print(df[target_col].value_counts(normalize=True))
    elif target_col:
        print(f"\nWarning: Target column '{target_col}' not found for distribution analysis.")

def perform_advanced_eda(df, target_col, numeric_features, categorical_features, dataset_name, corr_threshold=0.0):
    """Performs advanced EDA, printing numerical summaries and identifying highly correlated features."""
    print("\n--- Advanced Exploratory Data Analysis (EDA) ---")
    features_to_drop = [] # Initialize list of features to drop based on correlation
    if df is None:
        print("DataFrame not available for Advanced EDA.")
        return
    if not target_col or target_col not in df.columns:
        print("Target column missing or invalid for Advanced EDA.")
        return

    print(f"\n=== Feature Type Summary ({dataset_name.upper()}) ===")
    print(f"Number of numeric features identified: {len(numeric_features)}")
    print(f"Number of categorical features identified: {len(categorical_features)}")

    # Calculate and print correlation summary for numeric features
    if numeric_features:
        # Ensure we only use valid numeric features present in the df
        valid_numeric_features_corr = [f for f in numeric_features if f in df.columns]
        if valid_numeric_features_corr:
            print("\nCalculating correlations for numeric features...")
            # Calculate correlation on the original data (handle potential NaNs if necessary)
            # Using pairwise deletion by default with .corr()
            try:
                corr = df[valid_numeric_features_corr].corr().abs() # Use absolute correlation
                # Select upper triangle of correlation matrix
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                # Find index of feature columns with correlation greater than threshold
                if corr_threshold > 0.0:
                    features_to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
                    print(f"\n=== Features to drop based on correlation threshold > {corr_threshold} ===")
                    if features_to_drop:
                        print(f"Identified {len(features_to_drop)} features to drop: {features_to_drop}")
                    else:
                        print("No features identified for dropping based on the threshold.")
                else:
                    print("\nCorrelation-based feature selection disabled (threshold=0.0).")

                # Print summary of high correlations (e.g., > 0.8) regardless of threshold for info
                print("\n=== High Correlation Summary (Absolute > 0.8) for Information ===")
                # Use the original (non-absolute) correlation matrix for unstacking if needed for display
                corr_display = df[valid_numeric_features_corr].corr()
                high_corr_info = corr_display.unstack().sort_values(kind="quicksort", ascending=False).drop_duplicates()
                high_corr_info = high_corr_info[(high_corr_info.abs() > 0.8) & (high_corr_info != 1.0)]
                if not high_corr_info.empty:
                    print(high_corr_info.to_string(float_format="{:.3f}".format))
                else:
                    print("No feature pairs with absolute correlation > 0.8 found.")

            except Exception as e:
                print(f"Could not calculate or print correlation summary/features to drop: {e}")
        else:
            print("\nSkipping correlation summary: No valid numeric features found.")
    else:
        print("\nSkipping correlation summary: No numeric features identified.")

    # Note: Other advanced EDA steps like detailed outlier analysis or
    # feature interaction summaries could be added here as numerical outputs
    # if needed, but keeping it focused on correlation as requested.

    return features_to_drop # Return the list of features to potentially drop


# --- Outlier Removal ---
def remove_outliers_iqr(df, numeric_features, factor=1.5):
    """Removes outliers from numeric features using the IQR method."""
    print(f"\n--- Removing Outliers using IQR (Factor={factor}) ---")
    if df is None or not numeric_features:
        print("DataFrame or numeric features missing, skipping outlier removal.")
        return df

    df_cleaned = df.copy()
    initial_rows = df_cleaned.shape[0]
    outlier_indices = set()

    print("Identifying outliers in numeric features...")
    for col in numeric_features:
        if col not in df_cleaned.columns:
            print(f"  Warning: Column '{col}' not found for outlier removal.")
            continue
        # Ensure column is numeric before calculating quantiles
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            # Find indices of outliers for this column
            col_outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)].index
            outlier_indices.update(col_outliers)
            # print(f"  Found {len(col_outliers)} outliers in '{col}'") # Optional: verbose logging
        else:
            print(f"  Warning: Column '{col}' is not numeric, skipping outlier check.")

    num_outliers_found = len(outlier_indices)
    if num_outliers_found > 0:
        print(f"Identified {num_outliers_found} rows with outliers across numeric features.")
        df_cleaned = df_cleaned.drop(index=list(outlier_indices))
        print(f"Removed {initial_rows - df_cleaned.shape[0]} rows containing outliers.")
        print(f"Dataset shape after outlier removal: {df_cleaned.shape}")
    else:
        print("No outliers found based on the IQR method.")

    return df_cleaned


# --- Data Preparation (Feature Identification) ---
def identify_features(df, target_col):
    """Identifies features (X), target (y), and feature types."""
    print("\n--- Identifying Features and Target ---")
    if df is None or target_col is None or target_col not in df.columns:
        print("Error: DataFrame or target column is missing/invalid for feature identification.")
        return None, None, None, None

    try:
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Identify numeric and categorical features
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        print(f"Target column: '{target_col}'")
        print(f"Identified {X.shape[1]} features.")
        print(f"  - {len(numeric_features)} numeric features.")
        print(f"  - {len(categorical_features)} categorical features.")

        return X, y, numeric_features, categorical_features

    except Exception as e:
        print(f"Error during feature identification: {e}")
        traceback.print_exc()
        return None, None, None, None


# --- Model Training & Evaluation Functions ---
# Note: Evaluation functions now take 'preprocessor' as an argument
# and the dataset_name suffix is used more consistently for output files.

def train_evaluate_base_random_forest(preprocessor, X_train, y_train, X_test, y_test, output_dir, dataset_name_suffix):
    """Trains and evaluates a base RandomForestClassifier pipeline (no tuning)."""
    print(f"\n--- Training and Evaluating Base RandomForestClassifier ({dataset_name_suffix.upper()}) ---")
    try:
        # Build a pipeline with the provided preprocessor and a base RandomForestClassifier
        pipe_base_rf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100))
        ])

        # Train the base model
        print(f"Fitting Base RandomForest pipeline ({dataset_name_suffix.upper()})...")
        start_time_train = time.time()
        pipe_base_rf.fit(X_train, y_train)
        end_time_train = time.time()
        print(f"Base RandomForest fitting completed in {end_time_train - start_time_train:.2f} seconds.")

        # Evaluate on the test set
        print(f"\n--- Evaluating Base RandomForest Model ({dataset_name_suffix.upper()}) on Test Set ---")
        start_time_pred = time.time()
        y_pred = pipe_base_rf.predict(X_test)
        end_time_pred = time.time()
        print(f"Prediction completed in {end_time_pred - start_time_pred:.2f} seconds.")

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Plot and save Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax_cm)
        ax_cm.set_title(f'Confusion Matrix ({dataset_name_suffix.upper()} - Base RF)')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_xlabel('Predicted Label')
        save_plot(fig_cm, f"confusion_matrix_{dataset_name_suffix}_base_rf.png", output_dir)

        # Plot and save ROC Curve (for binary classification)
        if len(np.unique(y_test)) == 2:
            try:
                y_proba = pipe_base_rf.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = roc_auc_score(y_test, y_proba)

                fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
                ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'ROC Curve ({dataset_name_suffix.upper()} - Base RF)')
                ax_roc.legend(loc="lower right")
                save_plot(fig_roc, f"roc_curve_{dataset_name_suffix}_base_rf.png", output_dir)
                print(f"ROC AUC Score: {roc_auc:.4f}")
            except Exception as roc_e:
                print(f"Could not generate/save Base RF ROC curve: {roc_e}")
        else:
            print("ROC curve skipped (not a binary classification task).")

    except Exception as e:
        print(f"Error during Base RandomForest training/evaluation: {e}")
        traceback.print_exc()


def train_evaluate_random_forest(preprocessor, X_train, y_train, X_test, y_test, output_dir, dataset_name_suffix):
    """Trains, tunes, and evaluates a RandomForestClassifier pipeline."""
    print(f"\n--- Training and Evaluating Tuned RandomForestClassifier ({dataset_name_suffix.upper()}) ---")
    best_model = None
    try:
        # Build a pipeline with the provided preprocessor and RandomForestClassifier
        pipe_rf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])

        # Define a parameter grid
        param_grid_rf = {
            'classifier__n_estimators': [100, 150],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 3]
        }

        cv_folds = 3
        print(f"Running GridSearchCV with {cv_folds} folds ({dataset_name_suffix.upper()})...")
        grid_search_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=cv_folds, n_jobs=-1, scoring='accuracy', verbose=1)

        start_time = time.time()
        grid_search_rf.fit(X_train, y_train)
        end_time = time.time()
        print(f"GridSearchCV completed in {end_time - start_time:.2f} seconds.")

        print("\nBest Parameters Found (RandomForest):", grid_search_rf.best_params_)
        print("Best Cross-validation Accuracy (RandomForest): {:.4f}".format(grid_search_rf.best_score_))
        best_model = grid_search_rf.best_estimator_

        # Evaluate on the test set
        print(f"\n--- Evaluating Tuned RandomForest Model ({dataset_name_suffix.upper()}) on Test Set ---")
        start_time_pred = time.time()
        y_pred = best_model.predict(X_test)
        end_time_pred = time.time()
        print(f"Prediction completed in {end_time_pred - start_time_pred:.2f} seconds.")
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Plot and save Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_title(f'Confusion Matrix ({dataset_name_suffix.upper()} - Tuned RF)')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_xlabel('Predicted Label')
        save_plot(fig_cm, f"confusion_matrix_{dataset_name_suffix}_tuned_rf.png", output_dir)

        # Plot and save ROC Curve (for binary classification)
        if len(np.unique(y_test)) == 2:
            try:
                y_proba = best_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = roc_auc_score(y_test, y_proba)

                fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
                ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'ROC Curve ({dataset_name_suffix.upper()} - Tuned RF)')
                ax_roc.legend(loc="lower right")
                save_plot(fig_roc, f"roc_curve_{dataset_name_suffix}_tuned_rf.png", output_dir)
                print(f"ROC AUC Score: {roc_auc:.4f}")
            except Exception as roc_e:
                print(f"Could not generate/save ROC curve: {roc_e}")
        else:
            print("ROC curve skipped (not a binary classification task).")

        # --- Generate and Save Learning Curve ---
        try:
            print(f"\nGenerating Learning Curve for Tuned RF ({dataset_name_suffix.upper()})...")
            # Need the actual classifier, not the pipeline for learning_curve's estimator param
            # And need the *processed* training data
            processed_X_train_lc = best_model.named_steps['preprocessor'].transform(X_train)
            actual_classifier_lc = best_model.named_steps['classifier']

            train_sizes, train_scores, test_scores = learning_curve(
                estimator=actual_classifier_lc, # Use the fitted classifier
                X=processed_X_train_lc,         # Use processed training data
                y=y_train,
                cv=3,                           # Use same CV folds as GridSearchCV for consistency
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5), # Use 5 points for the curve
                scoring='accuracy'
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            fig_lc, ax_lc = plt.subplots(figsize=(8, 5))
            ax_lc.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            ax_lc.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            ax_lc.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            ax_lc.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            ax_lc.set_title(f'Learning Curve ({dataset_name_suffix.upper()} - Tuned RF)')
            ax_lc.set_xlabel("Training examples")
            ax_lc.set_ylabel("Accuracy Score")
            ax_lc.legend(loc="best")
            ax_lc.grid(True)
            save_plot(fig_lc, f"learning_curve_{dataset_name_suffix}_tuned_rf.png", output_dir)
            print("Learning curve generated.")

        except Exception as lc_e:
            print(f"Could not generate/save learning curve: {lc_e}")
            traceback.print_exc()
            if 'fig_lc' in locals(): plt.close(fig_lc) # Close plot if error occurred


        # --- Save the best model ---
        if best_model:
            model_filename = f"tuned_rf_model_{dataset_name_suffix}.pkl"
            model_filepath = os.path.join(output_dir, model_filename)
            try:
                with open(model_filepath, 'wb') as f_model:
                    pickle.dump(best_model, f_model)
                print(f"\nBest Tuned RandomForest model saved to: {model_filepath}")
                model_size_bytes = os.path.getsize(model_filepath)
                model_size_mb = model_size_bytes / (1024 * 1024)
                print(f"Saved model size: {model_size_mb:.3f} MB")
            except Exception as save_e:
                print(f"\nError saving model to {model_filepath}: {save_e}")

    except Exception as e:
        print(f"\nError during Tuned RandomForest training/evaluation: {e}")
        traceback.print_exc()

    return best_model # Return the best model


def train_evaluate_pca_random_forest(preprocessor, X_train, y_train, X_test, y_test, output_dir, dataset_name_suffix):
    """Tunes PCA n_components and evaluates the best PCA + Base RandomForestClassifier pipeline."""
    # NOTE: This function uses the *provided* preprocessor (assumed to be scaled)
    print(f"\n--- Tuning PCA Components + Evaluating PCA + Base RandomForestClassifier ({dataset_name_suffix.upper()}) ---")
    try:
        # Determine the number of features after preprocessing to set PCA n_components
        try:
            X_train_processed_check = preprocessor.transform(X_train[:1]) # Transform a small sample
            n_features_after_preprocessing = X_train_processed_check.shape[1]
        except Exception as preproc_err:
             print(f"Warning: Could not determine exact feature count after preprocessing: {preproc_err}")
             n_features_after_preprocessing = len(preprocessor.transformers_[0][2]) + 50 # Fallback estimate
             print(f"Using estimated feature count: {n_features_after_preprocessing}")

        # --- Tune n_components for PCA ---
        max_possible_components = min(n_features_after_preprocessing, X_train.shape[0])
        n_components_range = [n for n in [5, 10, 15, 20] if n > 0 and n < max_possible_components]
        if not n_components_range and max_possible_components > 0:
             n_components_range = [min(5, max_possible_components)]

        if not n_components_range:
             print("Warning: Not enough features/samples for PCA tuning. Skipping PCA+RF.")
             return

        print(f"\nTuning PCA n_components in range: {n_components_range}...")
        best_n_components = -1
        best_accuracy = -1.0
        best_pca_pipeline = None
        pca_tuning_results = []

        for n in n_components_range:
            print(f"  Testing PCA with n_components={n}...")
            pca = PCA(n_components=n)
            rf_clf_pca = RandomForestClassifier(random_state=42, class_weight='balanced') # Base RF
            pca_pipeline_test = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('pca', pca),
                ('classifier', rf_clf_pca)
            ])

            start_fit_time = time.time()
            pca_pipeline_test.fit(X_train, y_train)
            fit_time = time.time() - start_fit_time

            start_pred_time = time.time()
            y_pred_test = pca_pipeline_test.predict(X_test)
            pred_time = time.time() - start_pred_time

            accuracy_test = accuracy_score(y_test, y_pred_test)
            print(f"    n={n}: Accuracy={accuracy_test:.4f}, Fit Time={fit_time:.2f}s, Pred Time={pred_time:.2f}s")
            pca_tuning_results.append({'n_components': n, 'accuracy': accuracy_test, 'fit_time': fit_time, 'pred_time': pred_time})

            if accuracy_test > best_accuracy:
                best_accuracy = accuracy_test
                best_n_components = n
                best_pca_pipeline = pca_pipeline_test

        print(f"\n--- Best PCA Configuration Found ---")
        print(f"Best n_components: {best_n_components}")
        print(f"Best Accuracy: {best_accuracy:.4f}")

        if pca_tuning_results:
             results_df_pca = pd.DataFrame(pca_tuning_results)
             print("\nPCA Tuning Results Summary:")
             print(results_df_pca.sort_values(by='accuracy', ascending=False))

        # --- Evaluate the Best PCA + RandomForest Model ---
        if best_pca_pipeline:
            print(f"\n--- Evaluating Best PCA (n={best_n_components}) + RandomForest Model on Test Set ---")
            start_pred_time_best = time.time()
            y_pred_best_pca = best_pca_pipeline.predict(X_test)
            end_pred_time_best = time.time()
            print(f"Prediction time for best model: {end_pred_time_best - start_pred_time_best:.2f}s")

            print(f"Accuracy (already calculated): {best_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_best_pca, zero_division=0))

            cm_pca_best = confusion_matrix(y_test, y_pred_best_pca)
            print("\nConfusion Matrix:")
            print(cm_pca_best)

            fig_cm_pca, ax_cm_pca = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_pca_best, annot=True, fmt='d', cmap='Greens', ax=ax_cm_pca)
            ax_cm_pca.set_title(f'Confusion Matrix ({dataset_name_suffix.upper()} - PCA n={best_n_components} + RF)')
            ax_cm_pca.set_ylabel('True Label')
            ax_cm_pca.set_xlabel('Predicted Label')
            save_plot(fig_cm_pca, f"confusion_matrix_{dataset_name_suffix}_pca_rf_best_n{best_n_components}.png", output_dir)

            if len(np.unique(y_test)) == 2:
                try:
                    y_proba_pca_best = best_pca_pipeline.predict_proba(X_test)[:, 1]
                    fpr_pca, tpr_pca, _ = roc_curve(y_test, y_proba_pca_best)
                    roc_auc_pca = roc_auc_score(y_test, y_proba_pca_best)

                    fig_roc_pca, ax_roc_pca = plt.subplots(figsize=(7, 5))
                    ax_roc_pca.plot(fpr_pca, tpr_pca, label=f'ROC curve (AUC = {roc_auc_pca:.2f})')
                    ax_roc_pca.plot([0, 1], [0, 1], 'k--')
                    ax_roc_pca.set_xlabel('False Positive Rate')
                    ax_roc_pca.set_ylabel('True Positive Rate')
                    ax_roc_pca.set_title(f'ROC Curve ({dataset_name_suffix.upper()} - PCA n={best_n_components} + RF)')
                    ax_roc_pca.legend(loc="lower right")
                    save_plot(fig_roc_pca, f"roc_curve_{dataset_name_suffix}_pca_rf_best_n{best_n_components}.png", output_dir)
                    print(f"ROC AUC Score: {roc_auc_pca:.4f}")
                except Exception as roc_e:
                    print(f"Could not generate/save best PCA+RF ROC curve: {roc_e}")
            else:
                print("ROC curve skipped (not a binary classification task).")
        else:
             print("Could not determine the best PCA pipeline from tuning.")

    except Exception as e:
        print(f"\nError during PCA + RandomForest pipeline: {e}")
        traceback.print_exc()


def train_evaluate_ann_smote(preprocessor, X_train, y_train, X_test, y_test, output_dir, dataset_name_suffix):
    """Trains and evaluates an ANN model with SMOTE oversampling."""
    # NOTE: This function uses the *provided* preprocessor (assumed to be scaled)
    print(f"\n--- Training and Evaluating ANN + SMOTE ({dataset_name_suffix.upper()}) ---")
    if not TF_AVAILABLE:
        print("Skipping ANN + SMOTE: TensorFlow is not available.")
        return

    try:
        # 1. Preprocess training and test data *separately* using the fitted preprocessor
        print("Preprocessing data for ANN...")
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        input_dim_ann = X_train_processed.shape[1]
        print(f"Input dimension for ANN: {input_dim_ann}")

        y_train_numeric = pd.to_numeric(y_train, errors='coerce').fillna(0).astype(int)
        y_test_numeric = pd.to_numeric(y_test, errors='coerce').fillna(0).astype(int)

        if not (len(np.unique(y_train_numeric)) == 2 and len(np.unique(y_test_numeric)) == 2):
             print("Warning: Target variable does not appear to be binary after conversion. ANN+SMOTE assumes binary classification.")

        # 2. Apply SMOTE only to the processed training data
        print("Applying SMOTE to the training data...")
        smote = SMOTE(random_state=42)
        print(f"Shape before SMOTE: {X_train_processed.shape}, Class distribution: {np.bincount(y_train_numeric)}")
        start_time_smote = time.time()
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train_numeric)
        end_time_smote = time.time()
        print(f"SMOTE completed in {end_time_smote - start_time_smote:.2f} seconds.")
        print(f"Shape after SMOTE: {X_train_balanced.shape}, Class distribution: {np.bincount(y_train_balanced)}")

        # 3. Define and Compile the ANN model structure
        def create_ann_model(input_shape):
            model = Sequential([
                Dense(64, input_dim=input_shape, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

        ann_model = create_ann_model(input_dim_ann)
        print("\nANN Model Summary:")
        ann_model.summary(print_fn=lambda x: print(x))

        # 4. Train the ANN model on balanced data
        print("\nTraining ANN model...")
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        start_time_ann = time.time()
        history = ann_model.fit(
            X_train_balanced, y_train_balanced,
            validation_split=0.2,
            epochs=30,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )
        end_time_ann = time.time()
        print(f"ANN training completed in {end_time_ann - start_time_ann:.2f} seconds.")

        # 5. Evaluate the model on the *original* processed test set
        print(f"\n--- Evaluating ANN + SMOTE Model ({dataset_name_suffix.upper()}) on Test Set ---")
        start_time_pred = time.time()
        y_pred_prob_ann = ann_model.predict(X_test_processed)
        y_pred_ann = (y_pred_prob_ann > 0.5).astype("int32").flatten()
        end_time_pred = time.time()
        print(f"Prediction completed in {end_time_pred - start_time_pred:.2f} seconds.")

        accuracy_ann = accuracy_score(y_test_numeric, y_pred_ann)
        print(f"Accuracy: {accuracy_ann:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_numeric, y_pred_ann, zero_division=0))

        cm_ann = confusion_matrix(y_test_numeric, y_pred_ann)
        print("\nConfusion Matrix:")
        print(cm_ann)

        fig_cm_ann, ax_cm_ann = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Purples', ax=ax_cm_ann)
        ax_cm_ann.set_title(f'Confusion Matrix ({dataset_name_suffix.upper()} - ANN+SMOTE)')
        ax_cm_ann.set_ylabel('True Label')
        ax_cm_ann.set_xlabel('Predicted Label')
        save_plot(fig_cm_ann, f"confusion_matrix_{dataset_name_suffix}_ann.png", output_dir)

        try:
            history_df = pd.DataFrame(history.history)
            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            history_df.plot(ax=ax_hist)
            ax_hist.set_title(f'ANN Training History ({dataset_name_suffix.upper()})')
            ax_hist.set_xlabel('Epoch')
            ax_hist.set_ylabel('Metric Value')
            ax_hist.grid(True)
            ax_hist.set_ylim(0, max(1.0, history_df['loss'].max() * 1.1))
            save_plot(fig_hist, f"training_history_{dataset_name_suffix}_ann.png", output_dir)
        except Exception as hist_e:
            print(f"Could not generate/save training history plot: {hist_e}")

        if ann_model:
            model_filename = f"ann_smote_model_{dataset_name_suffix}.keras"
            model_filepath = os.path.join(output_dir, model_filename)
            try:
                ann_model.save(model_filepath)
                print(f"\nANN model saved to: {model_filepath}")
                if os.path.isfile(model_filepath):
                    model_size_bytes = os.path.getsize(model_filepath)
                    model_size_mb = model_size_bytes / (1024 * 1024)
                    print(f"Saved model size: {model_size_mb:.3f} MB")
                elif os.path.isdir(model_filepath):
                     total_size = 0
                     for dirpath, dirnames, filenames in os.walk(model_filepath):
                          for f in filenames:
                               fp = os.path.join(dirpath, f)
                               if not os.path.islink(fp):
                                    total_size += os.path.getsize(fp)
                     model_size_mb = total_size / (1024 * 1024)
                     print(f"Saved model directory size: {model_size_mb:.3f} MB")
            except Exception as save_e:
                print(f"\nError saving ANN model to {model_filepath}: {save_e}")

    except Exception as e:
        print(f"\nError during ANN + SMOTE pipeline: {e}")
        traceback.print_exc()


def run_fuzzy_cmeans(X, y, numeric_features, output_dir, dataset_name_suffix):
    """Performs Fuzzy C-Means clustering and evaluates it."""
    # NOTE: This function implicitly uses scaled data as it scales internally
    print(f"\n--- Running Fuzzy C-Means Clustering ({dataset_name_suffix.upper()}) ---")
    if not FUZZY_AVAILABLE:
        print("Skipping Fuzzy C-Means: scikit-fuzzy library not found.")
        return
    if not numeric_features:
        print("Skipping Fuzzy C-Means: No numeric features identified.")
        return

    try:
        # 1. Select and preprocess only numeric features for FCM
        print("Preprocessing numeric data for FCM...")
        X_fcm = X[numeric_features].copy()

        if X_fcm.isnull().sum().sum() > 0:
            print("Imputing missing values in numeric features with median...")
            imputer_fcm = SimpleImputer(strategy='median')
            X_fcm[:] = imputer_fcm.fit_transform(X_fcm)

        scaler_fcm = StandardScaler()
        X_scaled_fcm = scaler_fcm.fit_transform(X_fcm)
        print(f"Scaled numeric data shape for FCM: {X_scaled_fcm.shape}")

        y_numeric = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
        unique_labels_in_y = np.unique(y_numeric)
        mode_res_default = mode(y_numeric)
        if isinstance(mode_res_default.mode, (np.ndarray, list)):
             default_label = mode_res_default.mode[0] if len(mode_res_default.mode) > 0 else 0
        else:
             default_label = mode_res_default.mode
        default_label = default_label if len(unique_labels_in_y) > 0 else 0

        # --- Run Base Fuzzy C-Means (n=2, m=2.0) ---
        print("\n--- Running Base Fuzzy C-Means (n=2, m=2.0) ---")
        base_n_clusters = 2
        base_m = 2.0
        try:
            start_time_fcm_base = time.time()
            cntr_base, u_base, _, _, _, _, fpc_base = fuzz.cluster.cmeans(
                X_scaled_fcm.T, c=base_n_clusters, m=base_m, error=0.005, maxiter=1000, init=None, seed=42
            )
            fcm_labels_base = np.argmax(u_base, axis=0)
            end_time_fcm_base = time.time()
            print(f"Base FCM clustering completed in {end_time_fcm_base - start_time_fcm_base:.2f} seconds.")

            cluster_to_label_base = {}
            for cluster in range(base_n_clusters):
                cluster_mask = (fcm_labels_base == cluster)
                if np.sum(cluster_mask) > 0:
                    mode_result = mode(y_numeric[cluster_mask])
                    if isinstance(mode_result.mode, (np.ndarray, list)):
                         majority_class = mode_result.mode[0] if len(mode_result.mode) > 0 else default_label
                    else:
                         majority_class = mode_result.mode
                    cluster_to_label_base[cluster] = majority_class
                else:
                    cluster_to_label_base[cluster] = default_label

            fcm_pred_base = np.array([cluster_to_label_base.get(label, default_label) for label in fcm_labels_base])

            print("\n=== Base Fuzzy C-Means Results ===")
            accuracy_fcm_base = accuracy_score(y_numeric, fcm_pred_base)
            print(f"Accuracy: {accuracy_fcm_base:.4f}")
            print("Classification Report:")
            print(classification_report(y_numeric, fcm_pred_base, zero_division=0))
            print("Confusion Matrix:")
            print(confusion_matrix(y_numeric, fcm_pred_base))
            if len(np.unique(fcm_labels_base)) > 1:
                 try:
                      sil_score_base = silhouette_score(X_scaled_fcm, fcm_labels_base)
                      print(f"Silhouette Score: {sil_score_base:.4f}")
                 except ValueError as sil_err:
                      print(f"Warning: Could not compute silhouette score for base FCM: {sil_err}")
            else:
                 print("Silhouette Score not computable for base FCM (only one cluster found).")

        except Exception as e_base:
            print(f"Error during Base Fuzzy C-Means run: {e_base}")

        # 2. Fuzzy C-Means Clustering (with hyperparameter tuning)
        print("\n--- Running Tuned Fuzzy C-Means ---")
        n_clusters_range = [2, 3, 4]
        m_range = [1.5, 2.0, 2.5]
        best_accuracy_fcm = -1.0
        best_params_fcm = (None, None)
        best_fcm_pred = None
        best_fcm_labels = None
        results_fcm = []

        print("\nTuning FCM parameters (n_clusters, m)...")
        for n in n_clusters_range:
            for m_val in m_range:
                print(f"  Trying n_clusters={n}, m={m_val}...")
                try:
                    start_time_fcm_iter = time.time()
                    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                        X_scaled_fcm.T, c=n, m=m_val, error=0.005, maxiter=1000, init=None, seed=42
                    )
                    fcm_labels = np.argmax(u, axis=0)
                    end_time_fcm_iter = time.time()
                    print(f"    FCM iteration took {end_time_fcm_iter - start_time_fcm_iter:.2f} seconds.")

                    cluster_to_label = {}
                    for cluster in range(n):
                        cluster_mask = (fcm_labels == cluster)
                        if np.sum(cluster_mask) > 0:
                            mode_result = mode(y_numeric[cluster_mask])
                            if isinstance(mode_result.mode, (np.ndarray, list)):
                                 majority_class = mode_result.mode[0] if len(mode_result.mode) > 0 else default_label
                            else:
                                 majority_class = mode_result.mode
                            cluster_to_label[cluster] = majority_class
                        else:
                            cluster_to_label[cluster] = default_label

                    fcm_pred = np.array([cluster_to_label.get(label, default_label) for label in fcm_labels])

                    accuracy_fcm = accuracy_score(y_numeric, fcm_pred)
                    sil_score = -1.0
                    if len(np.unique(fcm_labels)) > 1:
                         try:
                              sil_score = silhouette_score(X_scaled_fcm, fcm_labels)
                         except ValueError as sil_err:
                              print(f"    Warning: Could not compute silhouette score for n={n}, m={m_val}: {sil_err}")

                    results_fcm.append({'n_clusters': n, 'm': m_val, 'accuracy': accuracy_fcm, 'silhouette': sil_score, 'fpc': fpc})
                    print(f"    Accuracy: {accuracy_fcm:.4f}, Silhouette: {sil_score:.4f}, FPC: {fpc:.4f}")

                    if accuracy_fcm > best_accuracy_fcm:
                        best_accuracy_fcm = accuracy_fcm
                        best_params_fcm = (n, m_val)
                        best_fcm_pred = fcm_pred
                        best_fcm_labels = fcm_labels

                except Exception as e_inner:
                    print(f"    Error during Fuzzy C-Means inner loop for n={n}, m={m_val}: {e_inner}")

        print("\n--- Best Fuzzy C-Means Results (from Tuning) ---")
        if best_fcm_pred is not None:
            print(f"Best Parameters Found: n_clusters={best_params_fcm[0]}, m={best_params_fcm[1]}")
            print(f"Best Accuracy (from tuning evaluation): {best_accuracy_fcm:.4f}")
            print("\nClassification Report (Best FCM):")
            print(classification_report(y_numeric, best_fcm_pred, zero_division=0))
            print("\nConfusion Matrix:")
            cm_fcm = confusion_matrix(y_numeric, best_fcm_pred)
            print(cm_fcm)

            if best_fcm_labels is not None and len(np.unique(best_fcm_labels)) > 1:
                try:
                    sil_score_best = silhouette_score(X_scaled_fcm, best_fcm_labels)
                    print(f"Silhouette Score (Best): {sil_score_best:.4f}")
                except ValueError as sil_err:
                    print(f"Warning: Could not compute silhouette score for best model: {sil_err}")
            else:
                print("Silhouette Score not computable for best model.")

            fig_cm_fcm, ax_cm_fcm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_fcm, annot=True, fmt='d', cmap='YlGnBu', ax=ax_cm_fcm)
            ax_cm_fcm.set_title(f'Confusion Matrix ({dataset_name_suffix.upper()} - Best FCM: n={best_params_fcm[0]}, m={best_params_fcm[1]})')
            ax_cm_fcm.set_ylabel('True Label')
            ax_cm_fcm.set_xlabel('Predicted Label (via Majority Vote)')
            save_plot(fig_cm_fcm, f"confusion_matrix_{dataset_name_suffix}_fcm.png", output_dir)

        else:
            print("Could not determine best FCM configuration from tuning.")

        if results_fcm:
            results_df_fcm = pd.DataFrame(results_fcm)
            print("\nFCM Tuning Results Summary:")
            print(results_df_fcm.sort_values(by='accuracy', ascending=False))

            if best_params_fcm[0] is not None:
                 try:
                      print(f"\nRe-running FCM with best params (n={best_params_fcm[0]}, m={best_params_fcm[1]}) to get centers...")
                      best_cntr, _, _, _, _, _, _ = fuzz.cluster.cmeans(
                           X_scaled_fcm.T, c=best_params_fcm[0], m=best_params_fcm[1], error=0.005, maxiter=1000, init=None, seed=42
                      )
                      centers_filename = f"tuned_fcm_centers_{dataset_name_suffix}_n{best_params_fcm[0]}_m{best_params_fcm[1]:.1f}.pkl"
                      centers_filepath = os.path.join(output_dir, centers_filename)
                      with open(centers_filepath, 'wb') as f_centers:
                           pickle.dump(best_cntr, f_centers)
                      print(f"Best Tuned Fuzzy C-Means centers saved to: {centers_filepath}")
                      centers_size_bytes = os.path.getsize(centers_filepath)
                      centers_size_mb = centers_size_bytes / (1024 * 1024)
                      print(f"Saved centers file size: {centers_size_mb:.6f} MB")
                 except Exception as save_e:
                      print(f"\nError saving FCM centers: {save_e}")

    except Exception as e:
        print(f"\nError during Fuzzy C-Means analysis: {e}")
        traceback.print_exc()


def run_shap_analysis(best_model, X_test, numeric_features, categorical_features, output_dir, dataset_name_suffix):
    """
    Performs SHAP analysis on the provided (pipeline) model.
    Returns a Pandas Series of feature importances (mean absolute SHAP values) or None if failed.
    """
    # NOTE: Assumes best_model contains a fitted preprocessor (likely scaled)
    print(f"\n--- Running SHAP Analysis ({dataset_name_suffix.upper()}) ---")
    feature_importance_sorted = None # Initialize return value
    if not SHAP_AVAILABLE:
        print("Skipping SHAP analysis: SHAP library not found.")
        return
    if best_model is None:
        print("Skipping SHAP analysis: No valid 'best_model' provided.")
        return
    if not isinstance(best_model, Pipeline):
         print("Skipping SHAP analysis: Provided model is not a scikit-learn Pipeline.")
         return

    try:
        # 1. Extract the fitted preprocessor and classifier from the pipeline
        try:
            fitted_preprocessor = best_model.named_steps['preprocessor']
            classifier_shap = best_model.named_steps['classifier']
        except KeyError:
            print("Error: Could not find 'preprocessor' or 'classifier' steps in the best_model pipeline.")
            return
        except AttributeError:
             print("Error: 'best_model' does not seem to be a Pipeline object.")
             return

        if not hasattr(classifier_shap, 'predict_proba') or not hasattr(classifier_shap, 'estimators_'):
             print(f"Skipping SHAP: Classifier type {type(classifier_shap)} might not be directly compatible with shap.TreeExplainer.")
             return

        # 2. Transform the test set using the *fitted* preprocessor
        print("Preprocessing test data for SHAP...")
        X_test_transformed_shap = fitted_preprocessor.transform(X_test)
        print(f"Shape of data transformed for SHAP: {X_test_transformed_shap.shape}")

        # 3. Get feature names after preprocessing
        processed_feature_names_shap = None
        try:
            ohe_transformer = fitted_preprocessor.named_transformers_['cat']['onehot']
            cat_feature_names_shap = ohe_transformer.get_feature_names_out(categorical_features)
            # Get numeric features used by the preprocessor
            numeric_features_in_preprocessor = fitted_preprocessor.transformers_[0][2] # Assumes 'num' is first
            processed_feature_names_shap = list(numeric_features_in_preprocessor) + list(cat_feature_names_shap)
            print(f"Number of features for SHAP: {len(processed_feature_names_shap)}")

            if len(processed_feature_names_shap) != X_test_transformed_shap.shape[1]:
                 print(f"Warning: Mismatch between generated feature names ({len(processed_feature_names_shap)}) and transformed data columns ({X_test_transformed_shap.shape[1]}). SHAP plot labels might be incorrect.")
                 processed_feature_names_shap = [f'feature_{i}' for i in range(X_test_transformed_shap.shape[1])]

        except Exception as e:
            print(f"Warning: Could not retrieve feature names for SHAP: {e}. Using generic names.")
            num_processed_features = X_test_transformed_shap.shape[1]
            processed_feature_names_shap = [f'feature_{i}' for i in range(num_processed_features)]

        X_test_transformed_df_shap = pd.DataFrame(X_test_transformed_shap, columns=processed_feature_names_shap, index=X_test.index) # Keep index

        # 4. Create SHAP Explainer for Tree models
        print("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(classifier_shap)

        # 5. Calculate SHAP values (use a smaller sample for speed)
        sample_size_shap = min(100, X_test_transformed_df_shap.shape[0])
        if sample_size_shap <= 0:
             print("Skipping SHAP: Not enough samples in the test set.")
             return

        X_sample_shap = shap.sample(X_test_transformed_df_shap, sample_size_shap, random_state=42)
        print(f"Calculating SHAP values for {sample_size_shap} samples...")
        start_time_shap = time.time()
        shap_values = explainer.shap_values(X_sample_shap)
        end_time_shap = time.time()
        print(f"SHAP calculation took {end_time_shap - start_time_shap:.2f} seconds.")

        shap_values_for_plot = None
        plot_title_base = f"SHAP Summary Plot ({dataset_name_suffix.upper()})"
        plot_title = plot_title_base

        if isinstance(shap_values, list) and len(shap_values) >= 2:
            print("SHAP values format: List of arrays per class.")
            shap_values_for_plot = shap_values[1]
            plot_title = f"{plot_title_base} (Class 1)"
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[-1] >= 2:
             print("SHAP values format: Single array (samples, features, classes).")
             shap_values_for_plot = shap_values[:, :, 1]
             plot_title = f"{plot_title_base} (Class 1)"
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
             print("SHAP values format: Single array (samples, features) - assuming positive class.")
             shap_values_for_plot = shap_values
             plot_title = f"{plot_title_base} (Assumed Positive Class)"
        else:
            print(f"Error: Unexpected SHAP values structure. Shape: {np.shape(shap_values)}. Cannot generate plot.")
            return

        if shap_values_for_plot is not None:
            try:
                mean_abs_shap = np.abs(shap_values_for_plot).mean(axis=0)
                feature_importance = pd.Series(mean_abs_shap, index=X_sample_shap.columns)
                feature_importance_sorted = feature_importance.sort_values(ascending=False)

                print("\n=== Mean Absolute SHAP Values (Feature Importance) ===")
                print(feature_importance_sorted)
                print("\n=== Top 10 Most Important Features (SHAP) ===")
                print(feature_importance_sorted.head(10))
                # Importance calculation successful
            except Exception as imp_e:
                 print(f"Error calculating SHAP feature importance: {imp_e}")
                 feature_importance_sorted = None # Ensure None if calculation fails
        else:
             print("Could not calculate SHAP feature importance as SHAP values for plotting are missing.")
             feature_importance_sorted = None # Ensure None if calculation fails

        # --- Attempt Plotting ---
        print(f"\nGenerating {plot_title}...")
        if shap_values_for_plot is not None:
            try:
                fig_shap, ax_shap = plt.subplots()
                shap.summary_plot(shap_values_for_plot, X_sample_shap, show=False, plot_size=None)
                fig_shap = plt.gcf()
                fig_shap.suptitle(plot_title, y=1.0)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                save_plot(fig_shap, f"shap_summary_plot_{dataset_name_suffix}.png", output_dir)
                print("SHAP plot generation attempted.")
            except Exception as plot_err:
                print(f"Error generating SHAP summary plot (continuing anyway): {plot_err}")
                plt.close(fig_shap if 'fig_shap' in locals() else plt.gcf()) # Close potentially broken plot
        else:
            print("Skipping SHAP summary plot generation as SHAP values are not available.")

    except ImportError:
        print("SHAP library is required for this analysis but not installed.")
        feature_importance_sorted = None # Ensure None is returned
    except Exception as e:
        print(f"An error occurred during SHAP analysis: {e}")
        traceback.print_exc()

    return feature_importance_sorted # Return the calculated importances (or None)


# --- Main Execution ---

def train_evaluate_base_random_forest(preprocessor, X_train, y_train, X_test, y_test, output_dir, dataset_name):
    """Trains and evaluates a base RandomForestClassifier pipeline (no tuning)."""
    print(f"\n--- Training and Evaluating Base RandomForestClassifier ({dataset_name.upper()}) ---")
    try:
        # Build a pipeline with the preprocessor and a base RandomForestClassifier
        # Using class_weight='balanced' is still a good default
        pipe_base_rf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)) # Default n_estimators
        ])

        # Train the base model
        print("Fitting Base RandomForest pipeline...")
        start_time_train = time.time()
        pipe_base_rf.fit(X_train, y_train)
        end_time_train = time.time()
        print(f"Base RandomForest fitting completed in {end_time_train - start_time_train:.2f} seconds.")

        # Evaluate on the test set
        print("\n--- Evaluating Base RandomForest Model on Test Set ---")
        start_time_pred = time.time()
        y_pred = pipe_base_rf.predict(X_test)
        end_time_pred = time.time()
        print(f"Prediction completed in {end_time_pred - start_time_pred:.2f} seconds.")

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Plot and save Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax_cm) # Different color
        ax_cm.set_title(f'Confusion Matrix ({dataset_name.upper()} - Base RF)')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_xlabel('Predicted Label')
        save_plot(fig_cm, f"confusion_matrix_{dataset_name}_base_rf.png", output_dir)

        # Plot and save ROC Curve (for binary classification)
        if len(np.unique(y_test)) == 2:
            try:
                y_proba = pipe_base_rf.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = roc_auc_score(y_test, y_proba)

                fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
                ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'ROC Curve ({dataset_name.upper()} - Base RF)')
                ax_roc.legend(loc="lower right")
                save_plot(fig_roc, f"roc_curve_{dataset_name}_base_rf.png", output_dir)
                print(f"ROC AUC Score: {roc_auc:.4f}")
            except Exception as roc_e:
                print(f"Could not generate/save Base RF ROC curve: {roc_e}")
        else:
            print("ROC curve skipped (not a binary classification task).")

    except Exception as e:
        print(f"Error during Base RandomForest training/evaluation: {e}")
        traceback.print_exc()


def train_evaluate_random_forest(preprocessor, X_train, y_train, X_test, y_test, output_dir, dataset_name):
    """Trains, tunes, and evaluates a RandomForestClassifier pipeline."""
    print(f"\n--- Training and Evaluating RandomForestClassifier ({dataset_name.upper()}) ---")
    best_model = None
    try:
        # Build a pipeline with the preprocessor and RandomForestClassifier
        pipe_rf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            # Use class_weight='balanced' for potentially imbalanced datasets
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])

        # Define a parameter grid (adjust complexity based on dataset size/time constraints)
        # Reduced grid for potentially faster execution
        param_grid_rf = {
            'classifier__n_estimators': [100, 150],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 3]
        }

        # Use fewer CV folds if dataset is large or for quicker runs
        cv_folds = 3
        print(f"Running GridSearchCV with {cv_folds} folds...")
        grid_search_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=cv_folds, n_jobs=-1, scoring='accuracy', verbose=1)

        start_time = time.time()
        grid_search_rf.fit(X_train, y_train)
        end_time = time.time()
        print(f"GridSearchCV completed in {end_time - start_time:.2f} seconds.")

        print("\nBest Parameters Found (RandomForest):", grid_search_rf.best_params_)
        print("Best Cross-validation Accuracy (RandomForest): {:.4f}".format(grid_search_rf.best_score_))
        best_model = grid_search_rf.best_estimator_

        # Evaluate on the test set
        print("\n--- Evaluating Tuned RandomForest Model on Test Set ---")
        start_time_pred = time.time()
        y_pred = best_model.predict(X_test)
        end_time_pred = time.time()
        print(f"Prediction completed in {end_time_pred - start_time_pred:.2f} seconds.")
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        # Use zero_division=0 to avoid warnings when a class has no predicted samples
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Plot and save Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_title(f'Confusion Matrix ({dataset_name.upper()} - Tuned RF)')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_xlabel('Predicted Label')
        save_plot(fig_cm, "confusion_matrix_rf.png", output_dir)

        # Plot and save ROC Curve (for binary classification)
        # Check if target is binary based on unique values in y_test
        if len(np.unique(y_test)) == 2:
            try:
                y_proba = best_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = roc_auc_score(y_test, y_proba)

                fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
                ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'ROC Curve ({dataset_name.upper()} - Tuned RF)')
                ax_roc.legend(loc="lower right")
                save_plot(fig_roc, f"roc_curve_{dataset_name}_rf.png", output_dir)
                print(f"ROC AUC Score: {roc_auc:.4f}")
            except Exception as roc_e:
                print(f"Could not generate/save ROC curve: {roc_e}")
        else:
            print("ROC curve skipped (not a binary classification task).")

        # --- Save the best model ---
        if best_model:
            model_filename = f"tuned_rf_model_{dataset_name}.pkl"
            model_filepath = os.path.join(output_dir, model_filename)
            try:
                with open(model_filepath, 'wb') as f_model:
                    pickle.dump(best_model, f_model)
                print(f"\nBest Tuned RandomForest model saved to: {model_filepath}")
                # Report model size
                model_size_bytes = os.path.getsize(model_filepath)
                model_size_mb = model_size_bytes / (1024 * 1024)
                print(f"Saved model size: {model_size_mb:.3f} MB")
            except Exception as save_e:
                print(f"\nError saving model to {model_filepath}: {save_e}")

    except Exception as e:
        print(f"\nError during RandomForest training/evaluation: {e}")
        traceback.print_exc()

    return best_model # Return the best model for potential SHAP analysis


def train_evaluate_pca_random_forest(preprocessor, X_train, y_train, X_test, y_test, output_dir, dataset_name):
    """Tunes PCA n_components and evaluates the best PCA + Base RandomForestClassifier pipeline."""
    print(f"\n--- Tuning PCA Components + Evaluating PCA + Base RandomForestClassifier ({dataset_name.upper()}) ---")
    try:
        # Determine the number of features after preprocessing to set PCA n_components
        try:
            X_train_processed_check = preprocessor.transform(X_train[:1]) # Transform a small sample
            n_features_after_preprocessing = X_train_processed_check.shape[1]
        except Exception as preproc_err:
             print(f"Warning: Could not determine exact feature count after preprocessing: {preproc_err}")
             # Estimate based on numeric + a guess for categorical
             n_features_after_preprocessing = len(preprocessor.transformers_[0][2]) + 50 # Fallback estimate
             print(f"Using estimated feature count: {n_features_after_preprocessing}")

        # --- Tune n_components for PCA ---
        # Define range, ensuring max is not more than available features or samples
        max_possible_components = min(n_features_after_preprocessing, X_train.shape[0])
        # Ensure components are positive and less than max_possible_components
        n_components_range = [n for n in [5, 10, 15, 20] if n > 0 and n < max_possible_components]
        if not n_components_range and max_possible_components > 0: # Ensure at least one value if possible
             n_components_range = [min(5, max_possible_components)]

        if not n_components_range:
             print("Warning: Not enough features/samples for PCA tuning. Skipping PCA+RF.")
             return

        print(f"\nTuning PCA n_components in range: {n_components_range}...")
        best_n_components = -1
        best_accuracy = -1.0
        best_pca_pipeline = None
        pca_tuning_results = []

        for n in n_components_range:
            print(f"  Testing PCA with n_components={n}...")
            pca = PCA(n_components=n)
            rf_clf_pca = RandomForestClassifier(random_state=42, class_weight='balanced') # Base RF
            pca_pipeline_test = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('pca', pca),
                ('classifier', rf_clf_pca)
            ])

            # Fit and evaluate this specific n_components value
            start_fit_time = time.time()
            pca_pipeline_test.fit(X_train, y_train)
            fit_time = time.time() - start_fit_time

            start_pred_time = time.time()
            y_pred_test = pca_pipeline_test.predict(X_test)
            pred_time = time.time() - start_pred_time

            accuracy_test = accuracy_score(y_test, y_pred_test)
            print(f"    n={n}: Accuracy={accuracy_test:.4f}, Fit Time={fit_time:.2f}s, Pred Time={pred_time:.2f}s")
            pca_tuning_results.append({'n_components': n, 'accuracy': accuracy_test, 'fit_time': fit_time, 'pred_time': pred_time})

            if accuracy_test > best_accuracy:
                best_accuracy = accuracy_test
                best_n_components = n
                best_pca_pipeline = pca_pipeline_test # Store the best pipeline

        print(f"\n--- Best PCA Configuration Found ---")
        print(f"Best n_components: {best_n_components}")
        print(f"Best Accuracy: {best_accuracy:.4f}")

        # Print tuning summary
        if pca_tuning_results:
             results_df_pca = pd.DataFrame(pca_tuning_results)
             print("\nPCA Tuning Results Summary:")
             print(results_df_pca.sort_values(by='accuracy', ascending=False))

        # --- Evaluate the Best PCA + RandomForest Model ---
        if best_pca_pipeline:
            print(f"\n--- Evaluating Best PCA (n={best_n_components}) + RandomForest Model on Test Set ---")
            # Re-predict using the stored best pipeline (or could re-fit, but predict is faster)
            start_pred_time_best = time.time()
            y_pred_best_pca = best_pca_pipeline.predict(X_test)
            end_pred_time_best = time.time()
            print(f"Prediction time for best model: {end_pred_time_best - start_pred_time_best:.2f}s")

            print(f"Accuracy (already calculated): {best_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_best_pca, zero_division=0))

            # Confusion matrix
            cm_pca_best = confusion_matrix(y_test, y_pred_best_pca)
            print("\nConfusion Matrix:")
            print(cm_pca_best)

            # Plot and save Confusion Matrix for the best PCA config
            fig_cm_pca, ax_cm_pca = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_pca_best, annot=True, fmt='d', cmap='Greens', ax=ax_cm_pca)
            ax_cm_pca.set_title(f'Confusion Matrix ({dataset_name.upper()} - Best PCA n={best_n_components} + RF)')
            ax_cm_pca.set_ylabel('True Label')
            ax_cm_pca.set_xlabel('Predicted Label')
            save_plot(fig_cm_pca, f"confusion_matrix_pca_rf_best_n{best_n_components}.png", output_dir)

            # Plot and save ROC Curve for the best PCA config (binary classification)
            if len(np.unique(y_test)) == 2:
                try:
                    y_proba_pca_best = best_pca_pipeline.predict_proba(X_test)[:, 1]
                    fpr_pca, tpr_pca, _ = roc_curve(y_test, y_proba_pca_best)
                    roc_auc_pca = roc_auc_score(y_test, y_proba_pca_best)

                    fig_roc_pca, ax_roc_pca = plt.subplots(figsize=(7, 5))
                    ax_roc_pca.plot(fpr_pca, tpr_pca, label=f'ROC curve (AUC = {roc_auc_pca:.2f})')
                    ax_roc_pca.plot([0, 1], [0, 1], 'k--')
                    ax_roc_pca.set_xlabel('False Positive Rate')
                    ax_roc_pca.set_ylabel('True Positive Rate')
                    ax_roc_pca.set_title(f'ROC Curve ({dataset_name.upper()} - Best PCA n={best_n_components} + RF)')
                    ax_roc_pca.legend(loc="lower right")
                    save_plot(fig_roc_pca, f"roc_curve_{dataset_name}_pca_rf_best_n{best_n_components}.png", output_dir)
                    print(f"ROC AUC Score: {roc_auc_pca:.4f}")
                except Exception as roc_e:
                    print(f"Could not generate/save best PCA+RF ROC curve: {roc_e}")
            else:
                print("ROC curve skipped (not a binary classification task).")
        else:
             print("Could not determine the best PCA pipeline from tuning.")

    except Exception as e:
        print(f"\nError during PCA + RandomForest pipeline: {e}")
        traceback.print_exc()


def train_evaluate_ann_smote(preprocessor, X_train, y_train, X_test, y_test, output_dir, dataset_name):
    """Trains and evaluates an ANN model with SMOTE oversampling."""
    print(f"\n--- Training and Evaluating ANN + SMOTE ({dataset_name.upper()}) ---")
    if not TF_AVAILABLE:
        print("Skipping ANN + SMOTE: TensorFlow is not available.")
        return

    try:
        # 1. Preprocess training and test data *separately* using the fitted preprocessor
        print("Preprocessing data for ANN...")
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        input_dim_ann = X_train_processed.shape[1]
        print(f"Input dimension for ANN: {input_dim_ann}")

        # Ensure target variable is numeric for SMOTE and ANN
        # Convert y_train and y_test to numeric if they are not already
        y_train_numeric = pd.to_numeric(y_train, errors='coerce').fillna(0).astype(int) # Coerce errors to 0
        y_test_numeric = pd.to_numeric(y_test, errors='coerce').fillna(0).astype(int)

        # Check if target is binary after conversion
        if not (len(np.unique(y_train_numeric)) == 2 and len(np.unique(y_test_numeric)) == 2):
             print("Warning: Target variable does not appear to be binary after conversion. ANN+SMOTE assumes binary classification.")
             # You might want to return or handle multi-class differently here
             # For now, we'll proceed assuming the user intends binary classification

        # 2. Apply SMOTE only to the processed training data
        print("Applying SMOTE to the training data...")
        smote = SMOTE(random_state=42)
        print(f"Shape before SMOTE: {X_train_processed.shape}, Class distribution: {np.bincount(y_train_numeric)}")
        start_time_smote = time.time()
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train_numeric)
        end_time_smote = time.time()
        print(f"SMOTE completed in {end_time_smote - start_time_smote:.2f} seconds.")
        print(f"Shape after SMOTE: {X_train_balanced.shape}, Class distribution: {np.bincount(y_train_balanced)}")

        # 3. Define and Compile the ANN model structure
        def create_ann_model(input_shape):
            model = Sequential([
                Dense(64, input_dim=input_shape, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid') # Sigmoid for binary classification
            ])
            # Use binary_crossentropy for binary classification
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

        ann_model = create_ann_model(input_dim_ann)
        print("\nANN Model Summary:")
        ann_model.summary(print_fn=lambda x: print(x)) # Print summary to the log

        # 4. Train the ANN model on balanced data
        print("\nTraining ANN model...")
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        start_time_ann = time.time()
        history = ann_model.fit(
            X_train_balanced, y_train_balanced,
            validation_split=0.2, # Use part of the balanced training data for validation
            epochs=30,           # Reduced epochs for potentially faster run
            batch_size=64,
            callbacks=[early_stop],
            verbose=1            # Set to 1 or 2 for progress, 0 for silent
        )
        end_time_ann = time.time()
        print(f"ANN training completed in {end_time_ann - start_time_ann:.2f} seconds.")

        # 5. Evaluate the model on the *original* processed test set
        print("\n--- Evaluating ANN + SMOTE Model on Test Set ---")
        start_time_pred = time.time()
        y_pred_prob_ann = ann_model.predict(X_test_processed)
        y_pred_ann = (y_pred_prob_ann > 0.5).astype("int32").flatten() # Flatten for metrics
        end_time_pred = time.time()
        print(f"Prediction completed in {end_time_pred - start_time_pred:.2f} seconds.")

        accuracy_ann = accuracy_score(y_test_numeric, y_pred_ann)
        print(f"Accuracy: {accuracy_ann:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_numeric, y_pred_ann, zero_division=0))

        # Confusion Matrix
        cm_ann = confusion_matrix(y_test_numeric, y_pred_ann)
        print("\nConfusion Matrix:")
        print(cm_ann)

        # Plot and save Confusion Matrix
        fig_cm_ann, ax_cm_ann = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Purples', ax=ax_cm_ann) # Different color map
        ax_cm_ann.set_title(f'Confusion Matrix ({dataset_name.upper()} - ANN+SMOTE)')
        ax_cm_ann.set_ylabel('True Label')
        ax_cm_ann.set_xlabel('Predicted Label')
        save_plot(fig_cm_ann, "confusion_matrix_ann.png", output_dir)

        # Plot and save training history
        try:
            history_df = pd.DataFrame(history.history)
            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            history_df.plot(ax=ax_hist)
            ax_hist.set_title(f'ANN Training History ({dataset_name.upper()})')
            ax_hist.set_xlabel('Epoch')
            ax_hist.set_ylabel('Metric Value')
            ax_hist.grid(True)
            ax_hist.set_ylim(0, max(1.0, history_df['loss'].max() * 1.1)) # Adjust ylim dynamically
            save_plot(fig_hist, f"training_history_{dataset_name}_ann.png", output_dir)
        except Exception as hist_e:
            print(f"Could not generate/save training history plot: {hist_e}")

        # --- Save the trained ANN model ---
        if ann_model:
            model_filename = f"ann_smote_model_{dataset_name}.keras" # Use .keras format
            model_filepath = os.path.join(output_dir, model_filename)
            try:
                ann_model.save(model_filepath)
                print(f"\nANN model saved to: {model_filepath}")
                # Report model size (size of the .keras file/directory)
                if os.path.isfile(model_filepath):
                    model_size_bytes = os.path.getsize(model_filepath)
                    model_size_mb = model_size_bytes / (1024 * 1024)
                    print(f"Saved model size: {model_size_mb:.3f} MB")
                elif os.path.isdir(model_filepath): # SavedModel format creates a directory
                     total_size = 0
                     for dirpath, dirnames, filenames in os.walk(model_filepath):
                          for f in filenames:
                               fp = os.path.join(dirpath, f)
                               # skip if it is symbolic link
                               if not os.path.islink(fp):
                                    total_size += os.path.getsize(fp)
                     model_size_mb = total_size / (1024 * 1024)
                     print(f"Saved model directory size: {model_size_mb:.3f} MB")

            except Exception as save_e:
                print(f"\nError saving ANN model to {model_filepath}: {save_e}")

    except Exception as e:
        print(f"\nError during ANN + SMOTE pipeline: {e}")
        traceback.print_exc()


def run_fuzzy_cmeans(X, y, numeric_features, output_dir, dataset_name):
    """Performs Fuzzy C-Means clustering and evaluates it."""
    print(f"\n--- Running Fuzzy C-Means Clustering ({dataset_name.upper()}) ---")
    if not FUZZY_AVAILABLE:
        print("Skipping Fuzzy C-Means: scikit-fuzzy library not found.")
        return
    if not numeric_features:
        print("Skipping Fuzzy C-Means: No numeric features identified.")
        return

    try:
        # 1. Select and preprocess only numeric features for FCM
        print("Preprocessing numeric data for FCM...")
        X_fcm = X[numeric_features].copy()

        # Impute missing values (using median) before scaling
        if X_fcm.isnull().sum().sum() > 0:
            print("Imputing missing values in numeric features with median...")
            imputer_fcm = SimpleImputer(strategy='median')
            X_fcm[:] = imputer_fcm.fit_transform(X_fcm)

        # Scale the numeric features
        scaler_fcm = StandardScaler()
        X_scaled_fcm = scaler_fcm.fit_transform(X_fcm)
        print(f"Scaled numeric data shape for FCM: {X_scaled_fcm.shape}")

        # Ensure target variable is numeric for evaluation
        y_numeric = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
        unique_labels_in_y = np.unique(y_numeric)
        # Handle scalar mode result for default_label
        mode_res_default = mode(y_numeric)
        if isinstance(mode_res_default.mode, (np.ndarray, list)):
             default_label = mode_res_default.mode[0] if len(mode_res_default.mode) > 0 else 0
        else: # Scalar case
             default_label = mode_res_default.mode
        default_label = default_label if len(unique_labels_in_y) > 0 else 0


        # --- Run Base Fuzzy C-Means (n=2, m=2.0) ---
        print("\n--- Running Base Fuzzy C-Means (n=2, m=2.0) ---")
        base_n_clusters = 2
        base_m = 2.0
        try:
            start_time_fcm_base = time.time()
            cntr_base, u_base, _, _, _, _, fpc_base = fuzz.cluster.cmeans(
                X_scaled_fcm.T, c=base_n_clusters, m=base_m, error=0.005, maxiter=1000, init=None, seed=42
            )
            fcm_labels_base = np.argmax(u_base, axis=0)
            end_time_fcm_base = time.time()
            print(f"Base FCM clustering completed in {end_time_fcm_base - start_time_fcm_base:.2f} seconds.")

            # Map base cluster labels
            cluster_to_label_base = {}
            for cluster in range(base_n_clusters):
                cluster_mask = (fcm_labels_base == cluster)
                if np.sum(cluster_mask) > 0:
                    mode_result = mode(y_numeric[cluster_mask])
                    # Handle scalar mode result for majority_class
                    if isinstance(mode_result.mode, (np.ndarray, list)):
                         majority_class = mode_result.mode[0] if len(mode_result.mode) > 0 else default_label
                    else: # Scalar case
                         majority_class = mode_result.mode
                    cluster_to_label_base[cluster] = majority_class
                else:
                    cluster_to_label_base[cluster] = default_label # Assign default if cluster is empty

            fcm_pred_base = np.array([cluster_to_label_base.get(label, default_label) for label in fcm_labels_base])

            # Evaluate Base FCM
            print("\n=== Base Fuzzy C-Means Results ===")
            accuracy_fcm_base = accuracy_score(y_numeric, fcm_pred_base)
            print(f"Accuracy: {accuracy_fcm_base:.4f}")
            print("Classification Report:")
            print(classification_report(y_numeric, fcm_pred_base, zero_division=0))
            print("Confusion Matrix:")
            print(confusion_matrix(y_numeric, fcm_pred_base))
            if len(np.unique(fcm_labels_base)) > 1:
                 try:
                      sil_score_base = silhouette_score(X_scaled_fcm, fcm_labels_base)
                      print(f"Silhouette Score: {sil_score_base:.4f}")
                 except ValueError as sil_err:
                      print(f"Warning: Could not compute silhouette score for base FCM: {sil_err}")
            else:
                 print("Silhouette Score not computable for base FCM (only one cluster found).")

        except Exception as e_base:
            print(f"Error during Base Fuzzy C-Means run: {e_base}")
        # --- End Base Fuzzy C-Means ---


        # 2. Fuzzy C-Means Clustering (with hyperparameter tuning)
        print("\n--- Running Tuned Fuzzy C-Means ---")
        n_clusters_range = [2, 3, 4] # Tune number of clusters (start simple)
        m_range = [1.5, 2.0, 2.5]    # Tune fuzziness parameter
        best_accuracy_fcm = -1.0
        best_params_fcm = (None, None)
        best_fcm_pred = None
        best_fcm_labels = None
        results_fcm = []

        print("\nTuning FCM parameters (n_clusters, m)...")
        for n in n_clusters_range:
            for m_val in m_range:
                print(f"  Trying n_clusters={n}, m={m_val}...")
                try:
                    start_time_fcm_iter = time.time()
                    # Perform fuzzy clustering
                    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                        X_scaled_fcm.T, c=n, m=m_val, error=0.005, maxiter=1000, init=None, seed=42
                    )
                    fcm_labels = np.argmax(u, axis=0)  # Assign clusters based on max membership
                    end_time_fcm_iter = time.time()
                    print(f"    FCM iteration took {end_time_fcm_iter - start_time_fcm_iter:.2f} seconds.")

                    # Map cluster labels to actual class labels using majority voting
                    cluster_to_label = {}
                    # default_label is already defined above

                    for cluster in range(n):
                        cluster_mask = (fcm_labels == cluster)
                        if np.sum(cluster_mask) > 0:
                            mode_result = mode(y_numeric[cluster_mask])
                            # Handle scalar mode result for majority_class
                            if isinstance(mode_result.mode, (np.ndarray, list)):
                                 majority_class = mode_result.mode[0] if len(mode_result.mode) > 0 else default_label
                            else: # Scalar case
                                 majority_class = mode_result.mode
                            cluster_to_label[cluster] = majority_class
                        else:
                            # Handle empty cluster - assign a default label
                            cluster_to_label[cluster] = default_label

                    fcm_pred = np.array([cluster_to_label.get(label, default_label) for label in fcm_labels])

                    # Compute accuracy and silhouette score
                    accuracy_fcm = accuracy_score(y_numeric, fcm_pred)
                    # Silhouette score requires at least 2 unique cluster labels
                    sil_score = -1.0 # Default if not computable
                    if len(np.unique(fcm_labels)) > 1:
                         try:
                              sil_score = silhouette_score(X_scaled_fcm, fcm_labels)
                         except ValueError as sil_err:
                              print(f"    Warning: Could not compute silhouette score for n={n}, m={m_val}: {sil_err}")

                    results_fcm.append({'n_clusters': n, 'm': m_val, 'accuracy': accuracy_fcm, 'silhouette': sil_score, 'fpc': fpc})
                    print(f"    Accuracy: {accuracy_fcm:.4f}, Silhouette: {sil_score:.4f}, FPC: {fpc:.4f}")

                    if accuracy_fcm > best_accuracy_fcm:
                        best_accuracy_fcm = accuracy_fcm
                        best_params_fcm = (n, m_val)
                        best_fcm_pred = fcm_pred
                        best_fcm_labels = fcm_labels

                except Exception as e_inner:
                    print(f"    Error during Fuzzy C-Means inner loop for n={n}, m={m_val}: {e_inner}")

        # 3. Evaluate the best configuration from tuning
        print("\n--- Best Fuzzy C-Means Results (from Tuning) ---")
        if best_fcm_pred is not None:
            # Note: Prediction time for FCM is essentially the label mapping time,
            # which was already done during the tuning loop for the best model.
            # We can report the accuracy found during tuning.
            print(f"Best Parameters Found: n_clusters={best_params_fcm[0]}, m={best_params_fcm[1]}")
            print(f"Best Accuracy (from tuning evaluation): {best_accuracy_fcm:.4f}")
            # Re-print classification report and CM for clarity in the summary
            print("\nClassification Report (Best FCM):")
            print(classification_report(y_numeric, best_fcm_pred, zero_division=0))
            print("\nConfusion Matrix:")
            cm_fcm = confusion_matrix(y_numeric, best_fcm_pred)
            print(cm_fcm)

            # Calculate and print silhouette score for the best model
            if best_fcm_labels is not None and len(np.unique(best_fcm_labels)) > 1:
                try:
                    sil_score_best = silhouette_score(X_scaled_fcm, best_fcm_labels)
                    print(f"Silhouette Score (Best): {sil_score_best:.4f}")
                except ValueError as sil_err:
                    print(f"Warning: Could not compute silhouette score for best model: {sil_err}")
            else:
                print("Silhouette Score not computable for best model (only one cluster found or labels missing).")

            # Optional: Plot and save Confusion Matrix for best FCM
            fig_cm_fcm, ax_cm_fcm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_fcm, annot=True, fmt='d', cmap='YlGnBu', ax=ax_cm_fcm)
            ax_cm_fcm.set_title(f'Confusion Matrix ({dataset_name.upper()} - Best FCM: n={best_params_fcm[0]}, m={best_params_fcm[1]})')
            ax_cm_fcm.set_ylabel('True Label')
            ax_cm_fcm.set_xlabel('Predicted Label (via Majority Vote)')
            save_plot(fig_cm_fcm, f"confusion_matrix_{dataset_name}_fcm.png", output_dir)

        else:
            print("Could not determine best FCM configuration from tuning.")

        # Print tuning summary table
        if results_fcm:
            results_df_fcm = pd.DataFrame(results_fcm)
            print("\nFCM Tuning Results Summary:")
            print(results_df_fcm.sort_values(by='accuracy', ascending=False))

            # --- Save the best FCM centers ---
            if best_params_fcm[0] is not None: # Check if tuning found a best config
                 # Re-run FCM with best params to get the centers reliably
                 try:
                      print(f"\nRe-running FCM with best params (n={best_params_fcm[0]}, m={best_params_fcm[1]}) to get centers...")
                      best_cntr, _, _, _, _, _, _ = fuzz.cluster.cmeans(
                           X_scaled_fcm.T, c=best_params_fcm[0], m=best_params_fcm[1], error=0.005, maxiter=1000, init=None, seed=42
                      )
                      centers_filename = f"tuned_fcm_centers_{dataset_name}_n{best_params_fcm[0]}_m{best_params_fcm[1]:.1f}.pkl"
                      centers_filepath = os.path.join(output_dir, centers_filename)
                      with open(centers_filepath, 'wb') as f_centers:
                           pickle.dump(best_cntr, f_centers)
                      print(f"Best Tuned Fuzzy C-Means centers saved to: {centers_filepath}")
                      # Report centers file size
                      centers_size_bytes = os.path.getsize(centers_filepath)
                      centers_size_mb = centers_size_bytes / (1024 * 1024)
                      print(f"Saved centers file size: {centers_size_mb:.6f} MB") # Use more precision for small files
                 except Exception as save_e:
                      print(f"\nError saving FCM centers: {save_e}")

    except Exception as e:
        print(f"\nError during Fuzzy C-Means analysis: {e}")
        traceback.print_exc()


def run_shap_analysis(best_model, X_test, numeric_features, categorical_features, output_dir, dataset_name):
    """Performs SHAP analysis on the provided (pipeline) model."""
    print(f"\n--- Running SHAP Analysis ({dataset_name.upper()}) ---")
    if not SHAP_AVAILABLE:
        print("Skipping SHAP analysis: SHAP library not found.")
        return
    if best_model is None:
        print("Skipping SHAP analysis: No valid 'best_model' (RandomForest) provided.")
        return
    if not isinstance(best_model, Pipeline):
         print("Skipping SHAP analysis: Provided model is not a scikit-learn Pipeline.")
         return

    try:
        # 1. Extract the fitted preprocessor and classifier from the pipeline
        try:
            fitted_preprocessor = best_model.named_steps['preprocessor']
            classifier_shap = best_model.named_steps['classifier']
        except KeyError:
            print("Error: Could not find 'preprocessor' or 'classifier' steps in the best_model pipeline.")
            return
        except AttributeError:
             print("Error: 'best_model' does not seem to be a Pipeline object.")
             return

        # Ensure the classifier is compatible with TreeExplainer
        if not hasattr(classifier_shap, 'predict_proba') or not hasattr(classifier_shap, 'estimators_'):
             print(f"Skipping SHAP: Classifier type {type(classifier_shap)} might not be directly compatible with shap.TreeExplainer.")
             # Could potentially use KernelExplainer as a fallback, but it's much slower.
             return

        # 2. Transform the test set using the *fitted* preprocessor
        print("Preprocessing test data for SHAP...")
        X_test_transformed_shap = fitted_preprocessor.transform(X_test)
        print(f"Shape of data transformed for SHAP: {X_test_transformed_shap.shape}")

        # 3. Get feature names after preprocessing
        processed_feature_names_shap = None
        try:
            # Get feature names from the fitted preprocessor
            ohe_transformer = fitted_preprocessor.named_transformers_['cat']['onehot']
            cat_feature_names_shap = ohe_transformer.get_feature_names_out(categorical_features)
            # Use the original numeric_features list passed to the function
            processed_feature_names_shap = list(numeric_features) + list(cat_feature_names_shap)
            print(f"Number of features for SHAP: {len(processed_feature_names_shap)}")

            # Sanity check: Compare number of names to number of columns
            if len(processed_feature_names_shap) != X_test_transformed_shap.shape[1]:
                 print(f"Warning: Mismatch between generated feature names ({len(processed_feature_names_shap)}) and transformed data columns ({X_test_transformed_shap.shape[1]}). SHAP plot labels might be incorrect.")
                 # Fallback to generic names if mismatch
                 processed_feature_names_shap = [f'feature_{i}' for i in range(X_test_transformed_shap.shape[1])]

        except Exception as e:
            print(f"Warning: Could not retrieve feature names for SHAP: {e}. Using generic names.")
            num_processed_features = X_test_transformed_shap.shape[1]
            processed_feature_names_shap = [f'feature_{i}' for i in range(num_processed_features)]

        # Convert transformed data to DataFrame *with feature names* for SHAP
        # SHAP prefers DataFrames for plotting feature names correctly
        X_test_transformed_df_shap = pd.DataFrame(X_test_transformed_shap, columns=processed_feature_names_shap)

        # 4. Create SHAP Explainer for Tree models
        print("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(classifier_shap)

        # 5. Calculate SHAP values (use a smaller sample for speed)
        sample_size_shap = min(100, X_test_transformed_df_shap.shape[0]) # Sample 100 or fewer
        if sample_size_shap <= 0:
             print("Skipping SHAP: Not enough samples in the test set.")
             return

        X_sample_shap = shap.sample(X_test_transformed_df_shap, sample_size_shap, random_state=42)
        print(f"Calculating SHAP values for {sample_size_shap} samples...")
        start_time_shap = time.time()
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample_shap)
        end_time_shap = time.time()
        print(f"SHAP calculation took {end_time_shap - start_time_shap:.2f} seconds.")
        # print(f"Shape of shap_values: {np.shape(shap_values)}") # Debug print

        # --- Process SHAP values for plotting and importance ---
        # TreeExplainer for multi-class RandomForest returns a list [shap_class_0, shap_class_1, ...]
        # For binary classification, it returns [shap_class_0, shap_class_1]
        # We usually focus on the SHAP values for the positive class (class 1)
        shap_values_for_plot = None
        plot_title_base = f"SHAP Summary Plot ({dataset_name.upper()})"
        plot_title = plot_title_base # Default title

        if isinstance(shap_values, list) and len(shap_values) >= 2:
            print("SHAP values format: List of arrays per class.")
            shap_values_for_plot = shap_values[1] # Use class 1 for plotting importance
            plot_title = f"{plot_title_base} (Class 1)"
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[-1] >= 2:
             # Handles case where output is a single array (samples, features, classes)
             print("SHAP values format: Single array (samples, features, classes).")
             shap_values_for_plot = shap_values[:, :, 1] # Use class 1 slice
             plot_title = f"{plot_title_base} (Class 1)"
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
             # Handles case where output might be just for the positive class already
             print("SHAP values format: Single array (samples, features) - assuming positive class.")
             shap_values_for_plot = shap_values
             plot_title = f"{plot_title_base} (Assumed Positive Class)"
        else:
            print(f"Error: Unexpected SHAP values structure. Shape: {np.shape(shap_values)}. Cannot generate plot.")
            return # Cannot proceed

        # --- Calculate and Print SHAP Feature Importance ---
        if shap_values_for_plot is not None:
            try:
                # Calculate mean absolute SHAP values for feature importance
                mean_abs_shap = np.abs(shap_values_for_plot).mean(axis=0)
                feature_importance = pd.Series(mean_abs_shap, index=X_sample_shap.columns) # Use columns from the sample DataFrame
                feature_importance_sorted = feature_importance.sort_values(ascending=False)

                print("\n=== Mean Absolute SHAP Values (Feature Importance) ===")
                print(feature_importance_sorted)
                print("\n=== Top 10 Most Important Features (SHAP) ===")
                print(feature_importance_sorted.head(10))
            except Exception as imp_e:
                 print(f"Error calculating SHAP feature importance: {imp_e}")
        else:
             print("Could not calculate SHAP feature importance as SHAP values for plotting are missing.")


        # 6. Generate and save SHAP summary plot
        print(f"\nGenerating {plot_title}...")
        if shap_values_for_plot is not None:
            try:
                # Create a figure and axes for the plot
                fig_shap, ax_shap = plt.subplots()
                # Generate the plot on the provided axes
                shap.summary_plot(shap_values_for_plot, X_sample_shap, show=False, plot_size=None) # Let save_plot handle size
                # Get the current figure associated with the plot generated by shap
                fig_shap = plt.gcf()
                # Add title using suptitle - SHAP plot might clear existing titles
                fig_shap.suptitle(plot_title, y=1.0) # Adjust y position if needed
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
                save_plot(fig_shap, f"shap_summary_plot_{dataset_name}.png", output_dir)
            except Exception as plot_err:
                print(f"Error generating SHAP summary plot: {plot_err}")
                # Close the potentially empty figure if error occurred
                plt.close(fig_shap if 'fig_shap' in locals() else plt.gcf())
        else:
            print("Skipping SHAP summary plot generation as SHAP values are not available.")

    except ImportError:
        print("SHAP library is required for this analysis but not installed.")
    except Exception as e:
        print(f"An error occurred during SHAP analysis: {e}")
        traceback.print_exc()


# --- Modeling Pipeline Function ---
def run_modeling_steps(X, y, numeric_features, categorical_features, features_to_drop_corr, args, output_dir, run_suffix, skip_corr_selection=False, return_shap_importance=False):
    """
    Runs the core modeling steps: split, preprocess, train, evaluate.
    Can optionally skip correlation selection and optionally return SHAP importance.
    Returns: SHAP importance Series if return_shap_importance is True and SHAP runs successfully, else None.
    """
    dataset_base_name = args.dataset_name # Get base name for suffixes
    shap_importance_results = None # Initialize return value for SHAP importance

    # Split Data
    print(f"\n--- Splitting Data ({run_suffix}) (80% Train, 20% Test) ---")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print("Data split successfully.")
    except ValueError as e:
         if "The least populated class" in str(e):
              print(f"Stratify error during split: {e}. Trying without stratify.")
              X_train, X_test, y_train, y_test = train_test_split(
                   X, y, test_size=0.2, random_state=42
              )
              print("Split successful without stratify.")
         else:
              raise e # Re-raise other ValueErrors
    except Exception as e:
        print(f"Error during data splitting for {run_suffix}: {e}")
        traceback.print_exc()
        print(f"\nExecution stopped during data splitting for {run_suffix}.")
        # Cannot sys.exit here, maybe return an error flag? For now, print and continue if possible.
        return # Stop this run

    # --- Define and Fit Preprocessors ---
    # Unscaled Preprocessor
    numeric_transformer_unscaled = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer_unscaled = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor_unscaled = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_unscaled, numeric_features),
            ('cat', categorical_transformer_unscaled, categorical_features)],
        remainder='passthrough')

    # Scaled Preprocessor
    numeric_transformer_scaled = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer_scaled = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_scaled, numeric_features),
            ('cat', categorical_transformer_scaled, categorical_features)],
        remainder='passthrough')

    print(f"\nFitting UNSCALED preprocessor ({run_suffix})...")
    preprocessor_unscaled.fit(X_train)
    print("Unscaled preprocessor fitted.")
    print(f"\nFitting SCALED preprocessor ({run_suffix})...")
    preprocessor_scaled.fit(X_train)
    print("Scaled preprocessor fitted.")

    # --- Run models on UNSCALED data (RF only) ---
    print("\n" + "="*40)
    print(f" Model Training: UNSCALED Data ({run_suffix}) - RF Only")
    print("="*40)
    suffix_unscaled = f"{run_suffix}_unscaled"
    train_evaluate_base_random_forest(
         preprocessor_unscaled, X_train, y_train, X_test, y_test, output_dir, suffix_unscaled
    )
    train_evaluate_random_forest(
        preprocessor_unscaled, X_train, y_train, X_test, y_test, output_dir, suffix_unscaled
    )

    # --- Run models on SCALED data ---
    print("\n" + "="*40)
    print(f" Model Training: SCALED Data ({run_suffix})")
    print("="*40)
    suffix_scaled = f"{run_suffix}_scaled"
    best_rf_model_scaled = None

    # Base RF (Scaled)
    train_evaluate_base_random_forest(
         preprocessor_scaled, X_train, y_train, X_test, y_test, output_dir, suffix_scaled
    )
    # Tuned RF (Scaled)
    best_rf_model_scaled = train_evaluate_random_forest(
        preprocessor_scaled, X_train, y_train, X_test, y_test, output_dir, suffix_scaled
    )
    # PCA + Base RF (Scaled)
    train_evaluate_pca_random_forest(
         preprocessor_scaled, X_train, y_train, X_test, y_test, output_dir, suffix_scaled
    )
    # ANN + SMOTE (Scaled)
    if args.run_ann:
        if TF_AVAILABLE:
            train_evaluate_ann_smote(
                preprocessor_scaled, X_train, y_train, X_test, y_test, output_dir, suffix_scaled
            )
        else:
            print("\nSkipping ANN + SMOTE (Scaled): TensorFlow library not found.")
    else:
         print("\nSkipping ANN + SMOTE: Not requested via --run_ann flag.")

    # Fuzzy C-Means (Scaled)
    if args.run_fuzzy:
        if FUZZY_AVAILABLE:
             # Pass the full X, y for this run, FCM scales internally
             run_fuzzy_cmeans(X, y, numeric_features, output_dir, suffix_scaled)
        else:
             print("\nSkipping Fuzzy C-Means (Scaled): scikit-fuzzy library not found.")
    else:
         print("\nSkipping Fuzzy C-Means: Not requested via --run_fuzzy flag.")

    # SHAP Analysis (Based on Tuned SCALED RF)
    if args.run_shap:
        if SHAP_AVAILABLE and best_rf_model_scaled is not None:
             print(f"\n--- SHAP Analysis (Based on Tuned SCALED RF - {run_suffix}) ---")
             # Capture the returned importance series
             shap_importance_results = run_shap_analysis(
                 best_rf_model_scaled, X_test, numeric_features, categorical_features, output_dir, suffix_scaled
             )
        elif not SHAP_AVAILABLE:
             print("\nSkipping SHAP Analysis: SHAP library not found.")
        else:
             print("\nSkipping SHAP Analysis: Best Scaled RandomForest model was not trained successfully.")
    else:
         print("\nSkipping SHAP Analysis: Not requested via --run_shap flag.")

    # --- Run models on SELECTED features (SCALED) ---
    # Skip this section if explicitly told to, or if no correlation threshold set, or if no features were identified to drop
    if not skip_corr_selection and args.feature_selection_corr_threshold > 0.0 and features_to_drop_corr:
        print("\n" + "="*40)
        print(f" Model Training: Correlation Selected Features, SCALED ({run_suffix})")
        print(f" (Threshold > {args.feature_selection_corr_threshold})")
        print("="*40)

        print(f"\nDropping {len(features_to_drop_corr)} features based on initial correlation analysis...")
        valid_features_to_drop = [f for f in features_to_drop_corr if f in X_train.columns]
        if len(valid_features_to_drop) < len(features_to_drop_corr):
             print(f"Warning: Some features identified for dropping were not found in the current feature set: {list(set(features_to_drop_corr) - set(valid_features_to_drop))}")

        if valid_features_to_drop:
            X_train_selected = X_train.drop(columns=valid_features_to_drop, errors='ignore')
            X_test_selected = X_test.drop(columns=valid_features_to_drop, errors='ignore')
            print(f"X_train_selected shape: {X_train_selected.shape}")
            print(f"X_test_selected shape: {X_test_selected.shape}")

            numeric_features_sel = X_train_selected.select_dtypes(include=np.number).columns.tolist()
            categorical_features_sel = X_train_selected.select_dtypes(include=['object']).columns.tolist()

            print("\nFitting new SCALED preprocessor for selected features...")
            numeric_transformer_sel_scaled = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])
            categorical_transformer_sel_scaled = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            preprocessor_sel_scaled = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer_sel_scaled, numeric_features_sel),
                    ('cat', categorical_transformer_sel_scaled, categorical_features_sel)],
                remainder='passthrough')
            preprocessor_sel_scaled.fit(X_train_selected)
            print("Scaled preprocessor for selected features fitted.")

            suffix_sel_scaled = f"{run_suffix}_sel_scaled"
            train_evaluate_base_random_forest(
                 preprocessor_sel_scaled, X_train_selected, y_train, X_test_selected, y_test, output_dir, suffix_sel_scaled
            )
            train_evaluate_random_forest(
                preprocessor_sel_scaled, X_train_selected, y_train, X_test_selected, y_test, output_dir, suffix_sel_scaled
            )
        else:
                 print("\nNo valid features to drop after considering current feature set. Skipping runs on selected features.")

    elif args.feature_selection_corr_threshold > 0.0:
        print("\n" + "="*40)
        print(f" Correlation Feature Selection ({run_suffix})")
        print("="*40)
        print("\nNo features identified for dropping based on correlation threshold. Skipping runs on correlation selected features.")
    elif skip_corr_selection:
        print("\nSkipping Correlation-Based Feature Selection step as requested for this run.")

    # Return SHAP importance if requested and available
    if return_shap_importance:
        return shap_importance_results
    else:
        return None


# --- Main Execution ---
def main():
    """Main function to run the analysis pipeline."""
    args = parse_arguments()
    dataset_base_name = args.dataset_name # Store the original dataset name

    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    except OSError as e:
        print(f"Error creating output directory {args.output_dir}: {e}")
        sys.exit(1)

    # Define summary file path
    summary_filepath = os.path.join(args.output_dir, "summary.txt")

    # --- Start Logging to Summary File ---
    original_stdout = sys.stdout # Keep track of the original stdout
    with open(summary_filepath, 'w') as f_summary:
        # Redirect print statements to the summary file
        with redirect_stdout(f_summary):
            print("="*50)
            print(" Intrusion Detection System Analysis Report")
            print("="*50)
            print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Dataset: {args.dataset_name.upper()}")
            print(f"Input Directory: {args.input_dir}")
            print(f"Output Directory: {args.output_dir}")
            print(f"Sample Size: {'Full Dataset' if args.sample_size == 0 else args.sample_size}")
            print(f"Run ANN: {args.run_ann}")
            print(f"Run Fuzzy C-Means: {args.run_fuzzy}")
            print(f"Run SHAP: {args.run_shap}")
            # Removed --run_outlier_removal flag, now runs both scenarios
            print(f"Feature Selection Correlation Threshold: {args.feature_selection_corr_threshold if args.feature_selection_corr_threshold > 0 else 'Disabled'}")
            print("-"*50)

            # Load Data
            df_orig, target_col = load_data(dataset_base_name, args.input_dir, args.sample_size)
            if df_orig is None:
                print("\nExecution stopped due to data loading errors.", file=original_stdout)
                sys.exit(1)

            # --- Run Analysis BEFORE Outlier Removal ---
            print("\n" + "#"*60)
            print(" RUNNING ANALYSIS: BEFORE OUTLIER REMOVAL")
            print("#"*60)

            # Perform Basic EDA on original loaded data
            perform_basic_eda(df_orig, target_col)

            # Identify features on original data
            X_orig, y_orig, numeric_features_orig, categorical_features_orig = identify_features(df_orig, target_col)
            if X_orig is None:
                print("\nExecution stopped due to feature identification errors (Original Data).", file=original_stdout)
                sys.exit(1)

            # Perform Advanced EDA (Correlation analysis) on original data
            # This determines which features *might* be dropped later, regardless of outlier removal
            features_to_drop_corr = perform_advanced_eda(
                df_orig, target_col, numeric_features_orig, categorical_features_orig, dataset_base_name, args.feature_selection_corr_threshold
            )

            # Run the modeling steps on original data, requesting SHAP importance back
            shap_importance_orig = run_modeling_steps(
                X_orig, y_orig, numeric_features_orig, categorical_features_orig,
                features_to_drop_corr, args, args.output_dir, f"{dataset_base_name}_orig",
                return_shap_importance=True # Request SHAP importance
            )


            # --- Perform Outlier Removal ---
            df_cleaned = remove_outliers_iqr(df_orig, numeric_features_orig)

            # --- Run Analysis AFTER Outlier Removal (if any were removed) ---
            if df_cleaned.shape[0] < df_orig.shape[0]:
                print("\n" + "#"*60)
                print(" RUNNING ANALYSIS: AFTER OUTLIER REMOVAL")
                print("#"*60)

                # Perform Basic EDA on cleaned data
                perform_basic_eda(df_cleaned, target_col)

                # Identify features on cleaned data
                X_clean, y_clean, numeric_features_clean, categorical_features_clean = identify_features(df_cleaned, target_col)
                if X_clean is None:
                    print("\nExecution stopped due to feature identification errors (Cleaned Data).", file=original_stdout)
                    sys.exit(1) # Stop if feature identification fails on cleaned data

                # Run the modeling steps on cleaned data
                # Note: We still use features_to_drop_corr identified from the *original* data's correlations
                run_modeling_steps(X_clean, y_clean, numeric_features_clean, categorical_features_clean,
                                   features_to_drop_corr, args, args.output_dir, f"{dataset_base_name}_no_outliers")
            else:
                print("\n" + "#"*60)
                print(" SKIPPING ANALYSIS: AFTER OUTLIER REMOVAL (No outliers were removed)")
                print("#"*60)

            # --- Run Analysis using only TOP SHAP features (if requested and SHAP ran successfully) ---
            if args.run_shap_selection:
                if shap_importance_orig is not None and isinstance(shap_importance_orig, pd.Series):
                    print("\n" + "#"*60)
                    print(" RUNNING ANALYSIS: DYNAMIC TOP 10 SHAP FEATURES ONLY (from original data run)")
                    print("#"*60)

                    # Get top 10 processed feature names from SHAP results
                    top10_processed_names = shap_importance_orig.head(10).index.tolist()
                    print(f"Top 10 processed feature names from SHAP: {top10_processed_names}")

                    # Map processed names back to original column names
                    original_cols_for_shap = set()
                    for processed_name in top10_processed_names:
                        if processed_name in numeric_features_orig or processed_name in categorical_features_orig:
                            original_cols_for_shap.add(processed_name)
                        else:
                            # Attempt to map back from one-hot encoded name
                            # Assumes format like 'originalcol_value'
                            parts = processed_name.split('_')
                            potential_original = '_'.join(parts[:-1]) # Join all but last part
                            if potential_original in categorical_features_orig:
                                original_cols_for_shap.add(potential_original)
                            else:
                                # If direct match or prefix match fails, maybe it's numeric? Check again.
                                if processed_name in X_orig.columns:
                                     original_cols_for_shap.add(processed_name)
                                else:
                                     print(f"  Warning: Could not map SHAP feature '{processed_name}' back to an original column.")

                    original_cols_list = list(original_cols_for_shap)
                    print(f"Mapped back to {len(original_cols_list)} original features: {original_cols_list}")

                    if not original_cols_list:
                         print("Error: Could not identify any original columns from top SHAP features. Skipping SHAP selection run.")
                    else:
                        # Include target column
                        shap_cols_to_keep = original_cols_list + [target_col]
                        # Ensure columns exist in df_orig before selecting
                        shap_cols_to_keep = [col for col in shap_cols_to_keep if col in df_orig.columns]
                        df_shap_selected = df_orig[shap_cols_to_keep].copy()

                        print(f"Selected DataFrame shape for SHAP features: {df_shap_selected.shape}")
                        perform_basic_eda(df_shap_selected, target_col) # EDA on this subset

                        # Identify features for this subset
                        X_shap, y_shap, numeric_features_shap, categorical_features_shap = identify_features(df_shap_selected, target_col)

                        if X_shap is not None:
                            # Run modeling steps, skipping correlation selection
                            run_modeling_steps(X_shap, y_shap, numeric_features_shap, categorical_features_shap,
                                               features_to_drop_corr=None, # No correlation features to drop here
                                               args=args,
                                               output_dir=args.output_dir,
                                               run_suffix=f"{dataset_base_name}_shap_dynamic_top10",
                                               skip_corr_selection=True) # Explicitly skip corr selection step
                        else:
                            print("\nExecution stopped due to feature identification errors (SHAP Selected Data).", file=original_stdout)

                elif args.run_shap: # SHAP was requested but importance wasn't returned
                     print("\n" + "#"*60)
                     print(" SKIPPING DYNAMIC SHAP FEATURE SELECTION: SHAP analysis did not return importance values.")
                     print("#"*60)
                # else: SHAP wasn't run in the first place, so skip silently.

            else: # --run_shap_selection flag not set
                print("\nSkipping analysis run using only top SHAP features (--run_shap_selection not set).")


            print("\n" + "="*50)
            print(" FULL Analysis Complete")
            print("="*50)
            print(f"Results saved in: {args.output_dir}")
            print(f"Summary log: {summary_filepath}")

    # Print completion message to console as well
    print("\nAnalysis Complete.", file=original_stdout)
    print(f"Results saved in: {args.output_dir}", file=original_stdout)
    print(f"Summary log: {summary_filepath}", file=original_stdout)


if __name__ == "__main__":
    main()
