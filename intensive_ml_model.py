"""
Refactored Intensive ML Model Script with Timestamped Outputs

This script includes:
    - Enhanced feature extraction from URLs.
    - Data loading and preprocessing for a malicious URLs dataset.
    - Multiple machine learning model training and evaluation (LDA, Logistic Regression, SVM, Random Forest).
    - Energy and CO2 emissions tracking using CodeCarbon (if installed).
    - Saving all generated plots and emissions logs into a timestamped folder inside the "plots" directory.

Ensure the following packages are installed:
    - CodeCarbon (for tracking energy/CO2 metrics): pip install codecarbon
"""

import time
import os
import re
import math
import datetime
import pandas as pd  # type: ignore 
import numpy as np  # type: ignore 
import matplotlib.pyplot as plt  # type: ignore 
import seaborn as sns  # type: ignore 
from urllib.parse import urlparse  # type: ignore 
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore 
from sklearn.model_selection import train_test_split  # type: ignore 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # type: ignore 
from sklearn.decomposition import PCA  # type: ignore 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # type: ignore 
from sklearn.linear_model import LogisticRegression  # type: ignore 
from sklearn.svm import LinearSVC  # type: ignore 
from sklearn.ensemble import RandomForestClassifier  # type: ignore 
from sklearn.tree import plot_tree  # type: ignore 
from joblib import Parallel, delayed # type: ignore


import warnings

# Suppress FutureWarning messages matching a specific pattern
warnings.filterwarnings(
    "ignore",
    message=".*DataFrame concatenation with empty or all-NA entries.*",
    category=FutureWarning
)

try:
    from codecarbon import EmissionsTracker  # type: ignore 
except ImportError:
    print("Warning: CodeCarbon is not installed. Install it via pip for energy/CO2 tracking.")
    EmissionsTracker = None


# Utility Function to Save Plots
def save_plot(fig, filename, folder):
    """
    Save the figure to the specified folder with the given filename and close the figure.
    
    Parameters:
        fig (matplotlib.figure.Figure): The plot figure.
        filename (str): The file name to use (e.g., "plot.png").
        folder (str): The directory where the plot will be saved.
    """
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


# --- URL Feature Extraction ---
def extract_url_features(url):
    """
    Extract enhanced features from a URL string.
    
    Parameters:
        url (str): The URL to extract features from.
    
    Returns:
        dict: A dictionary containing various features derived from the URL.
    """
    features = {}
    parsed = urlparse(url)
    
    # Basic length-based features
    features['url_length'] = len(url)
    features['domain_length'] = len(parsed.netloc)
    features['path_length'] = len(parsed.path)
    
    # Count of numeric and special characters
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special'] = len(re.findall(r'[^A-Za-z0-9]', url))
    
    # Domain structure features
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_subdomains'] = len(parsed.netloc.split('.')) - 2  # Adjust as needed
    
    # URL path features
    features['num_slashes'] = url.count('/')
    features['num_params'] = len(parsed.params)
    features['has_query'] = int(bool(parsed.query))
    
    # Protocol and IP-based features
    features['uses_https'] = int(parsed.scheme == 'https')
    features['has_ip_address'] = int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc)))
    
    # Suspicious keywords check
    suspicious_words = ['login', 'secure', 'account', 'update', 'free', 'gift', 'verification']
    features['suspicious_words'] = int(any(word in url.lower() for word in suspicious_words))
    
    # Calculate URL entropy
    def calculate_entropy(s):
        prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(s)]
        entropy = -sum(p * math.log(p, 2) for p in prob)
        return entropy
    features['url_entropy'] = calculate_entropy(url)
    
    return features

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    """
    Load the dataset from a local directory, extract URL features in parallel using Joblib,
    and preprocess the data.
    
    Returns:
        tuple: (df, X, y, X_train, X_test, y_train, y_test, 
                X_train_scaled, X_test_scaled, scaler, label_encoder)
    """
    print("=== Data Loading and Preprocessing ===")
    start_time = time.time()
    
    # Define the local dataset directory and CSV file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "malicious_phish.csv")
    print("Using local dataset directory:", script_dir)
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    print("Dataset loaded. First few rows:\n", df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nClass distribution:")
    print(df['type'].value_counts())
    
    # Extract enhanced features from URLs in parallel using Joblib
    print("Extracting enhanced URL features using Joblib...")
    features_list = Parallel(n_jobs=-1, verbose=1)(delayed(extract_url_features)(url) for url in df['url'])
    X = pd.DataFrame(features_list)
    
    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(df['type'])
    
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features for algorithms that require scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    duration = time.time() - start_time
    print(f"Preprocessing completed in {duration:.4f} seconds.\n")
    
    return df, X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, le


# --- Data Visualization ---
def visualize_data(X, X_train, X_test, output_folder):
    """
    Produce visualizations to understand feature correlations and distributions,
    saving all plots to the specified output folder.
    
    Args:
        X (pd.DataFrame): Feature data.
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Test feature set.
        output_folder (str): Directory where plots will be saved.
    """
    print("=== Data Visualization ===")
    start_time = time.time()
    
    # Plot the feature correlation heatmap
    fig1 = plt.figure(figsize=(12, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    save_plot(fig1, "feature_correlation_matrix.png", output_folder)
    
    # Plot boxplots for feature distributions
    fig2 = plt.figure(figsize=(15, 10))
    X.boxplot()
    plt.xticks(rotation=45)
    plt.title('Feature Distributions')
    save_plot(fig2, "feature_distributions.png", output_folder)
    
    duration = time.time() - start_time
    print(f"Data visualization completed in {duration:.4f} seconds.\n")


# --- Model Training and Evaluation Functions ---
def run_lda_model(X_train, X_test, y_train, y_test, le, feature_names, output_folder):
    """
    Train and evaluate a Linear Discriminant Analysis (LDA) model.
    """
    print("=== LDA Model ===")
    start_time = time.time()
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"LDA Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion Matrix
    fig_cm = plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - LDA')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_plot(fig_cm, "lda_confusion_matrix.png", output_folder)
    
    # Feature Importance (absolute average coefficient values)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(lda.coef_).mean(axis=0)
    }).sort_values('Importance', ascending=False)
    fig_fi = plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance - LDA')
    save_plot(fig_fi, "lda_feature_importance.png", output_folder)
    
    # LDA 2D Transformation Visualization (if applicable)
    X_lda = lda.transform(X_train)
    if X_lda.shape[1] >= 2:
        fig_scatter = plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_train, cmap='viridis')
        plt.title('LDA Transformation of Training Data')
        plt.xlabel('First Discriminant')
        plt.ylabel('Second Discriminant')
        plt.colorbar(scatter, label='Classes')
        save_plot(fig_scatter, "lda_transformation.png", output_folder)
    
    duration = time.time() - start_time
    print(f"LDA model training and evaluation completed in {duration:.4f} seconds.\n")


def run_logistic_regression_model(X_train_scaled, X_test_scaled, y_train, y_test, le, feature_names, output_folder):
    """
    Train and evaluate a Logistic Regression model.
    """
    print("=== Logistic Regression Model ===")
    start_time = time.time()
    
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    y_pred = logreg.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Logistic Regression Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion Matrix
    fig_cm = plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_plot(fig_cm, "logreg_confusion_matrix.png", output_folder)
    
    # Feature Importance analysis using absolute coefficients
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(logreg.coef_).mean(axis=0)
    }).sort_values('Importance', ascending=False)
    fig_fi = plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance - Logistic Regression')
    save_plot(fig_fi, "logreg_feature_importance.png", output_folder)
    
    duration = time.time() - start_time
    print(f"Logistic Regression model training and evaluation completed in {duration:.4f} seconds.\n")


def run_svm_model(X_train_scaled, X_test_scaled, y_train, y_test, le, feature_names, output_folder):
    """
    Train and evaluate a linear SVM model (LinearSVC).
    """
    print("=== SVM Model ===")
    start_time = time.time()
    
    svm = LinearSVC(penalty='l2', loss='squared_hinge', dual=False,
                    C=1.0, tol=1e-4, max_iter=1000, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"LinearSVC Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion Matrix
    fig_cm = plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - LinearSVC')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_plot(fig_cm, "svm_confusion_matrix.png", output_folder)
    
    # Feature Importance analysis using absolute coefficients
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(svm.coef_).mean(axis=0)
    }).sort_values('Importance', ascending=False)
    fig_fi = plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance - LinearSVC')
    plt.xlabel('Average Absolute Coefficient Value')
    plt.tight_layout()
    save_plot(fig_fi, "svm_feature_importance.png", output_folder)
    
    duration = time.time() - start_time
    print(f"SVM model training and evaluation completed in {duration:.4f} seconds.\n")


def run_random_forest_model(X_train_scaled, X_test_scaled, y_train, y_test, le, feature_names, output_folder):
    """
    Train and evaluate a Random Forest Classifier model including additional visualizations.
    """
    print("=== Random Forest Classifier Model ===")
    start_time = time.time()
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion Matrix
    fig_cm = plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_plot(fig_cm, "rf_confusion_matrix.png", output_folder)
    
    # Feature Importance visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig_fi = plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    save_plot(fig_fi, "rf_feature_importance.png", output_folder)
    
    # Visualize a single decision tree from the Random Forest
    fig_tree = plt.figure(figsize=(20, 10))
    plot_tree(rf.estimators_[0], feature_names=feature_names, filled=True, max_depth=3, fontsize=10)
    plt.title("Random Forest Tree Visualization (First Tree, max_depth=3)")
    save_plot(fig_tree, "rf_tree_visualization.png", output_folder)
    
    # Visualize decision boundaries using PCA reduction to 2D
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Map the grid back to the original feature space using PCA inverse transform
    grid_original = pca.inverse_transform(grid_points)
    Z = rf.predict(grid_original).reshape(xx.shape)
    
    fig_db = plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=20)
    plt.title("Random Forest Decision Boundaries in PCA Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label='Classes')
    save_plot(fig_db, "rf_decision_boundaries.png", output_folder)
    
    duration = time.time() - start_time
    print(f"Random Forest model training and evaluation completed in {duration:.4f} seconds.\n")


# --- Main Execution ---
def main():
    """
    Main function to execute the machine learning pipeline.
    
    Steps:
        1. Define output directories with timestamp for unique run outputs.
        2. Initialize the CodeCarbon emissions tracker (if installed) with the run-specific output directory.
        3. Load, preprocess, and visualize the data.
        4. Train and evaluate multiple machine learning models.
        5. Report the overall execution time and CO2 emissions.
    """
    overall_start_time = time.time()
    
    # Define the base directory (script directory) and a main plots folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_folder = os.path.join(script_dir, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    # Create a timestamped subfolder for this specific run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(plots_folder, timestamp)
    os.makedirs(run_folder, exist_ok=True)
    print("Plots and logs will be saved to:", run_folder)
    
    # Initialize CodeCarbon emissions tracker if available, using the run_folder for output
    tracker = None
    if EmissionsTracker is not None:
        tracker = EmissionsTracker(
            project_name="Cumulative ML Model",
            measure_power_secs=1,
            log_level="error",
            output_dir=run_folder
        )
        tracker.start()
    else:
        print("Emissions tracking disabled due to missing CodeCarbon package.")
    
    # Load and preprocess data
    df, X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, le = load_and_preprocess_data()
    
    # Data visualization
    visualize_data(X, X_train, X_test, run_folder)
    
    # Model training and evaluation
    run_lda_model(X_train, X_test, y_train, y_test, le, X.columns, run_folder)
    run_logistic_regression_model(X_train_scaled, X_test_scaled, y_train, y_test, le, X.columns, run_folder)
    run_svm_model(X_train_scaled, X_test_scaled, y_train, y_test, le, X.columns, run_folder)
    run_random_forest_model(X_train_scaled, X_test_scaled, y_train, y_test, le, X.columns, run_folder)
    
    # Report overall metrics
    total_time = time.time() - overall_start_time
    if tracker is not None:
        total_co2 = tracker.stop()
        print(f"Total CO2 Emissions for entire run: {total_co2:.4f} kg")
    else:
        total_co2 = None
    
    print(f"Total execution time: {total_time:.4f} seconds")


if __name__ == '__main__':
    main()