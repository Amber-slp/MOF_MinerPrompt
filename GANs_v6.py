import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, \
    confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
import os
import json
from datetime import datetime
import joblib
from pathlib import Path

os.environ["JOBLIB_MULTIPROCESSING"] = "0"

warnings.filterwarnings('ignore')


# ==================== Set Random Seed ====================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(42)


# ==================== Data Loading and Preprocessing ====================
def load_and_preprocess_data(csv_path):
    """Load CSV and perform preprocessing"""
    df = pd.read_csv(csv_path)
    print(f"Original dataset size: {df.shape}")
    print(f"Column names: {df.columns.tolist()}\n")

    # Remove conductivity column and non-numeric columns
    cols_to_drop = []
    if 'conductivity' in df.columns:
        cols_to_drop.append('conductivity')

    # Find non-numeric columns (except target column conductivity_class)
    for col in df.columns:
        if col != 'conductivity_class' and col not in cols_to_drop:
            if df[col].dtype == 'object':
                cols_to_drop.append(col)

    if cols_to_drop:
        print(f"Removing columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Check target column
    if 'conductivity_class' not in df.columns:
        raise ValueError("Target column 'conductivity_class' not found")

    # Separate features and labels
    y = df['conductivity_class'].values
    X = df.drop(columns=['conductivity_class']).values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Number of features: {X.shape[1]}")
    print(f"Classes: {le.classes_}")
    print(f"Class distribution: {np.bincount(y_encoded)}\n")

    return X, y_encoded, le, df.drop(columns=['conductivity_class']).columns.tolist()


# ==================== Save Datasets ====================
def save_datasets(X_work, y_work, X_test, y_test, X_augmented, y_augmented,
                  label_encoder, feature_names, output_dir="dataset_output"):
    """
    Save the split datasets to the specified folder
    """
    # Create output directory [1,3](@ref)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Get current timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert labels back to original labels
    y_work_original = label_encoder.inverse_transform(y_work)
    y_test_original = label_encoder.inverse_transform(y_test)
    y_augmented_original = label_encoder.inverse_transform(y_augmented)

    # Create DataFrame and save test set
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['conductivity_class'] = y_test_original
    test_file = os.path.join(output_dir, f"test_set_{timestamp}.csv")
    test_df.to_csv(test_file, index=False)
    print(f"Test set saved: {test_file} (samples: {len(test_df)})")

    # Create DataFrame and save original work set
    work_df = pd.DataFrame(X_work, columns=feature_names)
    work_df['conductivity_class'] = y_work_original
    work_file = os.path.join(output_dir, f"original_work_set_{timestamp}.csv")
    work_df.to_csv(work_file, index=False)
    print(f"Original work set saved: {work_file} (samples: {len(work_df)})")

    # Create DataFrame and save augmented work set
    augmented_df = pd.DataFrame(X_augmented, columns=feature_names)
    augmented_df['conductivity_class'] = y_augmented_original
    augmented_file = os.path.join(output_dir, f"augmented_work_set_{timestamp}.csv")
    augmented_df.to_csv(augmented_file, index=False)
    print(f"Augmented work set saved: {augmented_file} (samples: {len(augmented_df)})")

    # Save metadata information
    metadata = {
        "timestamp": timestamp,
        "test_set_size": int(len(test_df)),
        "original_work_set_size": int(len(work_df)),
        "augmented_work_set_size": int(len(augmented_df)),
        "synthetic_samples_added": int(len(augmented_df) - len(work_df)),
        "feature_count": int(len(feature_names)),
        "class_names": [str(cls) for cls in label_encoder.classes_],
        "class_distribution_original": {
            "work_set": {str(cls): int(count) for cls, count in zip(label_encoder.classes_, np.bincount(y_work))},
            "test_set": {str(cls): int(count) for cls, count in zip(label_encoder.classes_, np.bincount(y_test))}
        }
    }

    metadata_file = os.path.join(output_dir, f"dataset_metadata_{timestamp}.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Dataset metadata saved: {metadata_file}")

    return test_file, work_file, augmented_file


# ==================== Create Model Save Directory Structure ====================
def create_model_directories(base_dir="saved_models"):
    """
    Create directory structure for model saving [1,2,3](@ref)
    """
    # Use pathlib to create directories (modern Python style) [1,3](@ref)
    base_path = Path(base_dir)

    # Define subdirectory structure
    subdirs = [
        "XGBoost",
        "LightGBM",
        "RandomForest",
        "GAN_Generator",
        "training_logs"
    ]

    # Create main directory and all subdirectories [3](@ref)
    base_path.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        subdir_path = base_path / subdir
        subdir_path.mkdir(exist_ok=True)
        print(f"Created directory: {subdir_path}")

    return base_path, subdirs


# ==================== Save Trained Models ====================
def save_models(results, generator, scaler, label_encoder, base_dir="saved_models"):
    """
    Save all trained models to corresponding subfolders [6,8](@ref)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory structure
    base_path, subdirs = create_model_directories(base_dir)

    print(f"\n{'=' * 60}")
    print("Starting model saving...")
    print(f"{'=' * 60}")

    # Save XGBoost model [8](@ref)
    if 'XGBoost' in results:
        xgb_dir = base_path / "XGBoost"
        xgb_model = results['XGBoost']['model']  # Need to include model object in results

        # Save model file
        xgb_model.save_model(xgb_dir / f"xgb_model_{timestamp}.json")

        # Save model metadata
        xgb_metadata = {
            "timestamp": timestamp,
            "model_type": "XGBoost",
            "accuracy": float(results['XGBoost']['accuracy']),
            "f1_weighted": float(results['XGBoost']['f1_weighted']),
            "feature_count": xgb_model.n_features_in_ if hasattr(xgb_model, 'n_features_in_') else "unknown"
        }

        with open(xgb_dir / f"metadata_{timestamp}.json", 'w') as f:
            json.dump(xgb_metadata, f, indent=2)

        print(f"✓ XGBoost model saved to: {xgb_dir}")

    # Save LightGBM model [8](@ref)
    if 'LightGBM' in results:
        lgb_dir = base_path / "LightGBM"
        lgb_model = results['LightGBM']['model']

        # Save model file
        joblib.dump(lgb_model, lgb_dir / f"lgb_model_{timestamp}.joblib")

        # Save model metadata
        lgb_metadata = {
            "timestamp": timestamp,
            "model_type": "LightGBM",
            "accuracy": float(results['LightGBM']['accuracy']),
            "f1_weighted": float(results['LightGBM']['f1_weighted']),
            "feature_count": lgb_model.n_features_in_ if hasattr(lgb_model, 'n_features_in_') else "unknown"
        }

        with open(lgb_dir / f"metadata_{timestamp}.json", 'w') as f:
            json.dump(lgb_metadata, f, indent=2)

        print(f"✓ LightGBM model saved to: {lgb_dir}")

    # Save Random Forest model [8](@ref)
    if 'RandomForest' in results:
        rf_dir = base_path / "RandomForest"
        rf_model = results['RandomForest']['model']

        # Save model file
        joblib.dump(rf_model, rf_dir / f"rf_model_{timestamp}.joblib")

        # Save model metadata
        rf_metadata = {
            "timestamp": timestamp,
            "model_type": "RandomForest",
            "accuracy": float(results['RandomForest']['accuracy']),
            "f1_weighted": float(results['RandomForest']['f1_weighted']),
            "feature_count": rf_model.n_features_in_ if hasattr(rf_model, 'n_features_in_') else "unknown"
        }

        with open(rf_dir / f"metadata_{timestamp}.json", 'w') as f:
            json.dump(rf_metadata, f, indent=2)

        print(f"✓ Random Forest model saved to: {rf_dir}")

    # Save GAN generator model [1,8](@ref)
    if generator is not None:
        gan_dir = base_path / "GAN_Generator"

        # Save PyTorch model [1](@ref)
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'timestamp': timestamp
        }, gan_dir / f"gan_generator_{timestamp}.pth")

        # Save scaler and metadata
        gan_metadata = {
            "timestamp": timestamp,
            "model_type": "GAN_Generator",
            "model_format": "PyTorch",
            "device": str(next(generator.parameters()).device) if generator is not None else "unknown"
        }

        # Save scaler
        if scaler is not None:
            joblib.dump(scaler, gan_dir / f"scaler_{timestamp}.joblib")

        # Save label encoder
        if label_encoder is not None:
            joblib.dump(label_encoder, gan_dir / f"label_encoder_{timestamp}.joblib")

        with open(gan_dir / f"metadata_{timestamp}.json", 'w') as f:
            json.dump(gan_metadata, f, indent=2)

        print(f"✓ GAN generator saved to: {gan_dir}")

    # Save training logs and overall results [1](@ref)
    logs_dir = base_path / "training_logs"

    # Save overall results summary
    summary = {
        "training_timestamp": timestamp,
        "models_saved": list(results.keys()) + (["GAN_Generator"] if generator is not None else []),
        "performance_summary": {
            model_name: {
                "accuracy": float(metrics['accuracy']),
                "f1_weighted": float(metrics['f1_weighted']),
                "f1_macro": float(metrics['f1_macro'])
            } for model_name, metrics in results.items()
        },
        "best_model": max(results.items(),
                          key=lambda x: x[1]['f1_weighted'])[0] if results else "none"
    }

    with open(logs_dir / f"training_summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Training logs saved to: {logs_dir}")

    print(f"\nAll models successfully saved to: {base_path}")
    return base_path


# ==================== GAN Model Definition ====================
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ==================== Balanced CGAN Training (with Early Stopping) ====================
def train_cgan_balanced(X_train, y_train, n_classes, epochs=3000, batch_size=16, noise_dim=100,
                        patience=300, min_delta=0.001):
    """
    Balanced discriminator and generator CGAN training with early stopping
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Data standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    feature_dim = X_scaled.shape[1]

    # Initialize models
    generator = Generator(noise_dim + n_classes, feature_dim, hidden_dim=256).to(device)
    discriminator = Discriminator(feature_dim + n_classes, hidden_dim=128).to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    # Convert to tensor
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)

    # Label smoothing and noise
    real_label_range = (0.8, 1.0)
    fake_label_range = (0.0, 0.2)

    print("Starting balanced CGAN training (with early stopping)...")
    g_losses = []
    d_losses = []

    # Early stopping related variables
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
