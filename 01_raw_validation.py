import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os


def analyze_raw_data(raw_data_path, output_dir='./results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- [步骤 1/3] 开始原始数据基准分析 ---")
    print(f"正在读取原始数据: {raw_data_path}")

    df = pd.read_csv(raw_data_path)

    # 1. 准备数据
    target_col = 'conductivity_class'

    # 排除非数值列和不相关的ID列 (如 Smiles)
    # 仅保留数值特征用于分析
    X = df.drop(columns=[target_col, 'conductivity', 'Smiles'], errors='ignore')
    X = X.select_dtypes(include=[np.number])

    y = df[target_col]

    # 编码目标变量
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        # 保存映射关系以便查看
        print(f"类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # 2. 训练基准随机森林 (Random Forest)
    # RF 对噪声鲁棒，适合作为 Feature Importance 的基准
    print("正在训练基准随机森林模型...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # 3. 提取特征重要性
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Raw_Importance': rf.feature_importances_
    }).sort_values(by='Raw_Importance', ascending=False)

    # 4. 保存结果
    output_file = os.path.join(output_dir, 'raw_data_importance.csv')
    importance_df.to_csv(output_file, index=False)

    print(f"原始数据分析完成！基准重要性已保存至: {output_file}")
    print(f"Top 3 原始特征: {importance_df.head(3)['Feature'].tolist()}\n")


if __name__ == "__main__":
    # 修改为你的实际路径
    RAW_PATH = r"rawdata/raw_v2.csv"
    analyze_raw_data(RAW_PATH)