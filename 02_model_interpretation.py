import pandas as pd
import numpy as np
import shap
import joblib
import os


def analyze_trained_model(model_path, data_path, output_dir='./results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- [步骤 2/3] 开始最终模型 SHAP 分析 ---")

    # 1. 加载模型
    print(f"加载模型: {os.path.basename(model_path)}")
    model = joblib.load(model_path)

    # 2. 加载增强后的训练数据 (必须与训练时特征一致)
    print(f"加载训练集数据: {os.path.basename(data_path)}")
    df = pd.read_csv(data_path)

    target_col = 'conductivity_class'
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df
    X = X.select_dtypes(include=[np.number])

    # 3. 计算 SHAP 值 (全类别)
    print("正在计算 SHAP 值 (涵盖 Class 0, 1, 2)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 注意：shap_values 是一个 list，包含 [array(Class0), array(Class1), array(Class2)]

    # 4. 打包保存
    # 我们保存所有需要绘图的信息
    analysis_pack = {
        "X": X,
        "shap_values_list": shap_values,  # 包含所有类别的 list
        "feature_names": X.columns.tolist()
    }

    output_file = os.path.join(output_dir, 'model_shap_data.joblib')
    joblib.dump(analysis_pack, output_file)

    print(f"模型解释完成！数据包已保存至: {output_file}\n")


if __name__ == "__main__":
    # 修改为你的实际路径
    MODEL_PATH = r"models/lgb_model_20251028_110128.joblib"
    DATA_PATH = r"set/augmented_work_set_20251028_123020.csv"

    analyze_trained_model(MODEL_PATH, DATA_PATH)