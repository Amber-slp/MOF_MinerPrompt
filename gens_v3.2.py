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

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(42)

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    cols_to_drop = []
    if 'conductivity' in df.columns:
        cols_to_drop.append('conductivity')

    for col in df.columns:
        if col != 'conductivity_class' and col not in cols_to_drop:
            if df[col].dtype == 'object':
                cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if 'conductivity_class' not in df.columns:
        raise ValueError("cannot find 'conductivity_class'")

    y = df['conductivity_class'].values
    X = df.drop(columns=['conductivity_class']).values


    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le, df.drop(columns=['conductivity_class']).columns.tolist()

def save_datasets(X_work, y_work, X_test, y_test, X_augmented, y_augmented,
                  label_encoder, feature_names, output_dir="dataset_output"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    y_work_original = label_encoder.inverse_transform(y_work)
    y_test_original = label_encoder.inverse_transform(y_test)
    y_augmented_original = label_encoder.inverse_transform(y_augmented)


    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['conductivity_class'] = y_test_original
    test_file = os.path.join(output_dir, f"test_set_{timestamp}.csv")
    test_df.to_csv(test_file, index=False)

    work_df = pd.DataFrame(X_work, columns=feature_names)
    work_df['conductivity_class'] = y_work_original
    work_file = os.path.join(output_dir, f"original_work_set_{timestamp}.csv")
    work_df.to_csv(work_file, index=False)



    augmented_df = pd.DataFrame(X_augmented, columns=feature_names)
    augmented_df['conductivity_class'] = y_augmented_original
    augmented_file = os.path.join(output_dir, f"augmented_work_set_{timestamp}.csv")
    augmented_df.to_csv(augmented_file, index=False)

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

    return test_file, work_file, augmented_file

def create_model_directories(base_dir="saved_models"):

    base_path = Path(base_dir)

    subdirs = [
        "XGBoost",
        "LightGBM",
        "RandomForest",
        "GAN_Generator",
        "training_logs"
    ]

    base_path.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        subdir_path = base_path / subdir
        subdir_path.mkdir(exist_ok=True)

    return base_path, subdirs

def save_models(results, generator, scaler, label_encoder, base_dir="saved_models"):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_path, subdirs = create_model_directories(base_dir)

    print(f"\n{'=' * 60}")
    print(f"{'=' * 60}")

    if 'XGBoost' in results:
        xgb_dir = base_path / "XGBoost"
        xgb_model = results['XGBoost']['model']

        xgb_model.save_model(xgb_dir / f"xgb_model_{timestamp}.json")

        xgb_metadata = {
            "timestamp": timestamp,
            "model_type": "XGBoost",
            "accuracy": float(results['XGBoost']['accuracy']),
            "f1_weighted": float(results['XGBoost']['f1_weighted']),
            "feature_count": xgb_model.n_features_in_ if hasattr(xgb_model, 'n_features_in_') else "unknown"
        }

        with open(xgb_dir / f"metadata_{timestamp}.json", 'w') as f:
            json.dump(xgb_metadata, f, indent=2)

        print(f"✓ XGBoost: {xgb_dir}")

    if 'LightGBM' in results:
        lgb_dir = base_path / "LightGBM"
        lgb_model = results['LightGBM']['model']


        joblib.dump(lgb_model, lgb_dir / f"lgb_model_{timestamp}.joblib")

        lgb_metadata = {
            "timestamp": timestamp,
            "model_type": "LightGBM",
            "accuracy": float(results['LightGBM']['accuracy']),
            "f1_weighted": float(results['LightGBM']['f1_weighted']),
            "feature_count": lgb_model.n_features_in_ if hasattr(lgb_model, 'n_features_in_') else "unknown"
        }

        with open(lgb_dir / f"metadata_{timestamp}.json", 'w') as f:
            json.dump(lgb_metadata, f, indent=2)

        print(f"✓ LightGBM: {lgb_dir}")

    if 'RandomForest' in results:
        rf_dir = base_path / "RandomForest"
        rf_model = results['RandomForest']['model']

        joblib.dump(rf_model, rf_dir / f"rf_model_{timestamp}.joblib")

        rf_metadata = {
            "timestamp": timestamp,
            "model_type": "RandomForest",
            "accuracy": float(results['RandomForest']['accuracy']),
            "f1_weighted": float(results['RandomForest']['f1_weighted']),
            "feature_count": rf_model.n_features_in_ if hasattr(rf_model, 'n_features_in_') else "unknown"
        }

        with open(rf_dir / f"metadata_{timestamp}.json", 'w') as f:
            json.dump(rf_metadata, f, indent=2)

        print(f"✓ Random Forest: {rf_dir}")

    if generator is not None:
        gan_dir = base_path / "GAN_Generator"

        torch.save({
            'generator_state_dict': generator.state_dict(),
            'timestamp': timestamp
        }, gan_dir / f"gan_generator_{timestamp}.pth")

        gan_metadata = {
            "timestamp": timestamp,
            "model_type": "GAN_Generator",
            "model_format": "PyTorch",
            "device": str(next(generator.parameters()).device) if generator is not None else "unknown"
        }

        if scaler is not None:
            joblib.dump(scaler, gan_dir / f"scaler_{timestamp}.joblib")

        if label_encoder is not None:
            joblib.dump(label_encoder, gan_dir / f"label_encoder_{timestamp}.joblib")

        with open(gan_dir / f"metadata_{timestamp}.json", 'w') as f:
            json.dump(gan_metadata, f, indent=2)

        print(f"✓ GAN: {gan_dir}")

    logs_dir = base_path / "training_logs"

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

    print(f"✓ login: {logs_dir}")

    print(f"\nall in: {base_path}")
    return base_path


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


def train_cgan_balanced(X_train, y_train, n_classes, epochs=3000, batch_size=16, noise_dim=100,
                        patience=300, min_delta=0.001):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}\n")


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    feature_dim = X_scaled.shape[1]


    generator = Generator(noise_dim + n_classes, feature_dim, hidden_dim=256).to(device)
    discriminator = Discriminator(feature_dim + n_classes, hidden_dim=128).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.000025, betas=(0.5, 0.999))

    criterion = nn.BCELoss()


    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)

    real_label_range = (0.8, 1.0)
    fake_label_range = (0.0, 0.2)

    g_losses = []
    d_losses = []

    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_generator_state = None
    loss_history = []

    for epoch in range(epochs):

        idx = np.random.randint(0, X_scaled.shape[0], min(batch_size, X_scaled.shape[0]))
        real_data = X_tensor[idx]
        real_labels = y_tensor[idx]
        current_batch_size = len(idx)

        real_labels_onehot = torch.zeros(current_batch_size, n_classes).to(device)
        real_labels_onehot.scatter_(1, real_labels.unsqueeze(1), 1)

        d_optimizer.zero_grad()


        real_label_value = torch.FloatTensor(current_batch_size, 1).uniform_(*real_label_range).to(device)
        real_input = torch.cat([real_data, real_labels_onehot], dim=1)
        real_output = discriminator(real_input)
        d_real_loss = criterion(real_output, real_label_value)

        noise = torch.randn(current_batch_size, noise_dim).to(device)
        fake_labels = torch.randint(0, n_classes, (current_batch_size,)).to(device)
        fake_labels_onehot = torch.zeros(current_batch_size, n_classes).to(device)
        fake_labels_onehot.scatter_(1, fake_labels.unsqueeze(1), 1)

        gen_input = torch.cat([noise, fake_labels_onehot], dim=1)
        fake_data = generator(gen_input)

        fake_label_value = torch.FloatTensor(current_batch_size, 1).uniform_(*fake_label_range).to(device)
        fake_input = torch.cat([fake_data.detach(), fake_labels_onehot], dim=1)
        fake_output = discriminator(fake_input)
        d_fake_loss = criterion(fake_output, fake_label_value)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        d_optimizer.step()

        for _ in range(2):
            g_optimizer.zero_grad()

            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_labels = torch.randint(0, n_classes, (current_batch_size,)).to(device)
            fake_labels_onehot = torch.zeros(current_batch_size, n_classes).to(device)
            fake_labels_onehot.scatter_(1, fake_labels.unsqueeze(1), 1)

            gen_input = torch.cat([noise, fake_labels_onehot], dim=1)
            fake_data = generator(gen_input)
            fake_input = torch.cat([fake_data, fake_labels_onehot], dim=1)
            fake_output = discriminator(fake_input)

            g_loss = criterion(fake_output, torch.ones(current_batch_size, 1).to(device))
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            g_optimizer.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        if (epoch + 1) % 50 == 0:
            avg_g_loss = np.mean(g_losses[-50:])
            avg_d_loss = np.mean(d_losses[-50:])

            combined_loss = avg_g_loss + avg_d_loss + abs(avg_g_loss - avg_d_loss)
            loss_history.append(combined_loss)

            if combined_loss < best_loss - min_delta:
                best_loss = combined_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_generator_state = generator.state_dict().copy()
                print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f} ✓ ")
            else:
                patience_counter += 50
                print(
                    f"Epoch [{epoch + 1}/{epochs}], D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f} ( {patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\nbestepoch: {best_epoch}, loss: {best_loss:.4f}")
                if best_generator_state is not None:
                    generator.load_state_dict(best_generator_state)
                break

        elif (epoch + 1) % 500 == 0:
            avg_g_loss = np.mean(g_losses[-100:])
            avg_d_loss = np.mean(d_losses[-100:])
            print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

    if patience_counter < patience:
        print(f"over\n")
    else:
        print(f"over{best_epoch}epoch\n")

    return generator, scaler, device, noise_dim

def generate_synthetic_data(generator, n_samples, n_classes, noise_dim, scaler, device, X_real):

    generator.eval()

    synthetic_X = []
    synthetic_y = []
    samples_per_class = n_samples // n_classes

    real_min = np.min(X_real, axis=0)
    real_max = np.max(X_real, axis=0)
    real_mean = np.mean(X_real, axis=0)
    real_std = np.std(X_real, axis=0)

    lower_bound = real_mean - 3 * real_std
    upper_bound = real_mean + 3 * real_std

    lower_bound = np.minimum(lower_bound, real_min)
    upper_bound = np.maximum(upper_bound, real_max)

    with torch.no_grad():
        for class_idx in range(n_classes):
            noise = torch.randn(samples_per_class, noise_dim).to(device)
            labels = torch.zeros(samples_per_class, n_classes).to(device)
            labels[:, class_idx] = 1

            gen_input = torch.cat([noise, labels], dim=1)
            fake_data = generator(gen_input).cpu().numpy()

            fake_data = scaler.inverse_transform(fake_data)
            for i in range(fake_data.shape[1]):
                fake_data[:, i] = np.clip(fake_data[:, i], lower_bound[i], upper_bound[i])

            synthetic_X.append(fake_data)
            synthetic_y.extend([class_idx] * samples_per_class)

    synthetic_X = np.vstack(synthetic_X)
    synthetic_y = np.array(synthetic_y)

    return synthetic_X, synthetic_y


def evaluate_generated_samples(generator, scaler, device, noise_dim, X_real, n_samples=1000):

    generator.eval()

    with torch.no_grad():
        real_min = np.min(X_real, axis=0)
        real_max = np.max(X_real, axis=0)
        real_mean = np.mean(X_real, axis=0)
        real_std = np.std(X_real, axis=0)

        lower_bound = real_mean - 4 * real_std
        upper_bound = real_mean + 4 * real_std

        all_generated = []
        valid_samples = 0
        total_attempts = 0

        while valid_samples < n_samples:
            for class_label in range(3):
                if valid_samples >= n_samples:
                    break

                noise = torch.randn(n_samples // 6, noise_dim).to(device)
                labels_onehot = torch.zeros(n_samples // 6, 3).to(device)
                labels_onehot[:, class_label] = 1

                gen_input = torch.cat([noise, labels_onehot], dim=1)
                fake_data = generator(gen_input).cpu().numpy()
                fake_data = scaler.inverse_transform(fake_data)

                valid_mask = np.ones(fake_data.shape[0], dtype=bool)
                for i in range(fake_data.shape[1]):
                    col_mask = (fake_data[:, i] >= lower_bound[i]) & (fake_data[:, i] <= upper_bound[i])
                    valid_mask &= col_mask

                valid_data = fake_data[valid_mask]

                if valid_data.shape[0] > 0:
                    all_generated.append(valid_data)
                    valid_samples += valid_data.shape[0]

                total_attempts += fake_data.shape[0]

        generated_samples = np.vstack(all_generated)

        validity_rate = valid_samples / total_attempts if total_attempts > 0 else 0
    return generated_samples


def compare_distributions(real_data, generated_data, feature_idx=0):

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(real_data[:, feature_idx], bins=50, alpha=0.7, label='Real')
    plt.hist(generated_data[:, feature_idx], bins=50, alpha=0.7, label='Generated')
    plt.legend()
    plt.title(f'Feature {feature_idx} Distribution')

    plt.subplot(1, 2, 2)
    plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.3, label='Real', s=10)
    plt.scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.3, label='Generated', s=10)
    plt.legend()
    plt.title('Feature Space Comparison')
    plt.show()


def train_classifiers(X_train, y_train, X_test, y_test, n_features='auto'):

    results = {}

    if n_features == 'auto':
        n_features = X_train.shape[1]
    else:
        n_features = min(n_features, X_train.shape[1])

    # ========== XGBoost ==========
    print("训练 XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        max_features=n_features
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    results['XGBoost'] = {
        'model': xgb_model,  # 保存模型对象
        'predictions': xgb_pred,
        'accuracy': accuracy_score(y_test, xgb_pred),
        'f1_macro': f1_score(y_test, xgb_pred, average='macro'),
        'f1_weighted': f1_score(y_test, xgb_pred, average='weighted'),
        'precision': precision_score(y_test, xgb_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, xgb_pred, average='weighted', zero_division=0)
    }

    # ========== LightGBM ==========
    print("训练 LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        feature_fraction=n_features / X_train.shape[1],
        verbose=-1,
        n_jobs=1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)

    results['LightGBM'] = {
        'model': lgb_model,  # 保存模型对象
        'predictions': lgb_pred,
        'accuracy': accuracy_score(y_test, lgb_pred),
        'f1_macro': f1_score(y_test, lgb_pred, average='macro'),
        'f1_weighted': f1_score(y_test, lgb_pred, average='weighted'),
        'precision': precision_score(y_test, lgb_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, lgb_pred, average='weighted', zero_division=0)
    }

    # ========== Random Forest ==========
    print("训练 Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        max_features=n_features
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    results['RandomForest'] = {
        'model': rf_model,  # 保存模型对象
        'predictions': rf_pred,
        'accuracy': accuracy_score(y_test, rf_pred),
        'f1_macro': f1_score(y_test, rf_pred, average='macro'),
        'f1_weighted': f1_score(y_test, rf_pred, average='weighted'),
        'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0)
    }

    return results


# ==================== 主函数 ====================
def main(csv_path, n_synthetic=10000, n_features=50, enable_visualization=True,
         patience=300, min_delta=0.001, output_dir="dataset_output", models_dir="saved_models"):
    """
    主流程
    """
    print("=" * 60)
    print("GAN数据增强 + 多分类器系统 (带模型保存功能)")
    print("=" * 60 + "\n")

    # 1. 加载数据
    X, y, label_encoder, feature_names = load_and_preprocess_data(csv_path)

    # 2. 分割预测集
    X_work, X_test, y_work, y_test = train_test_split(
        X, y, test_size=26, random_state=42, stratify=y
    )
    print(f"工作集大小: {X_work.shape[0]}")
    print(f"预测集大小: {X_test.shape[0]}\n")

    # 3. 训练CGAN（带早停）
    n_classes = len(np.unique(y))
    generator, scaler, device, noise_dim = train_cgan_balanced(
        X_work, y_work, n_classes, patience=patience, min_delta=min_delta
    )

    # 4. 生成合成数据
    X_synthetic, y_synthetic = generate_synthetic_data(
        generator, n_synthetic, n_classes, noise_dim, scaler, device, X_work
    )

    # 5. 验证生成质量
    print("=" * 60)
    print("生成样本质量评估")
    print("=" * 60)
    generated_samples = evaluate_generated_samples(
        generator, scaler, device, noise_dim, X_work, n_samples=1000
    )

    # 6. 可视化对比
    if enable_visualization and X_work.shape[1] >= 2:
        print("\n生成数据与真实数据对比可视化:")
        compare_distributions(X_work, generated_samples)

    # 7. 数据增强
    X_augmented = np.vstack([X_work, X_synthetic])
    y_augmented = np.hstack([y_work, y_synthetic])

    print(f"最终训练集大小: {X_augmented.shape[0]} (真实: {X_work.shape[0]}, 合成: {X_synthetic.shape[0]})\n")

    # 8. 保存数据集
    print("=" * 60)
    print("保存数据集")
    print("=" * 60)
    test_file, work_file, augmented_file = save_datasets(
        X_work, y_work, X_test, y_test, X_augmented, y_augmented,
        label_encoder, feature_names, output_dir
    )

    # 9. 训练分类器并评估
    print("\n" + "=" * 60)
    print("分类器训练与评估")
    print("=" * 60 + "\n")

    results = train_classifiers(X_augmented, y_augmented, X_test, y_test, n_features)

    # 10. 保存所有模型
    models_base_path = save_models(results, generator, scaler, label_encoder, models_dir)

    # 11. 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60 + "\n")

    for model_name, metrics in results.items():
        print(f"\n{'=' * 40}")
        print(f"  {model_name}")
        print(f"{'=' * 40}")
        print(f"准确率 (Accuracy):       {metrics['accuracy']:.4f}")
        print(f"F1分数 (Macro):          {metrics['f1_macro']:.4f}")
        print(f"F1分数 (Weighted):       {metrics['f1_weighted']:.4f}")
        print(f"精确率 (Precision):      {metrics['precision']:.4f}")
        print(f"召回率 (Recall):         {metrics['recall']:.4f}")

        print(f"\n详细分类报告:")
        print(classification_report(y_test, metrics['predictions'],
                                    target_names=label_encoder.classes_,
                                    zero_division=0))

    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['f1_weighted'])
    print(f"\n{'=' * 60}")
    print(f"最佳模型: {best_model[0]} (F1-Weighted: {best_model[1]['f1_weighted']:.4f})")
    print(f"{'=' * 60}\n")

    return results, generator, scaler, device, noise_dim, models_base_path


# ==================== 运行 ====================
if __name__ == "__main__":
    csv_path = "F:\mof_conduct\代码\GANS\特征选取\important_features_output.csv"

    results, generator, scaler, device, noise_dim, models_path = main(
        csv_path,
        n_synthetic=120,
        n_features=48,
        enable_visualization=False,
        patience=28,
        min_delta=0.001,
        output_dir="dataset_output/featurexiaorong",
        models_dir="saved_models/featurexiaorong"  # 指定模型保存文件夹
    )
