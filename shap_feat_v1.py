import pandas as pd
import numpy as np
import shap
import joblib
import os


def run_feature_analysis(model_path, data_path, output_dir='./results'):
    # 0. 路径检查与创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[1/4] 正在加载模型: {os.path.basename(model_path)}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    print(f"[2/4] 正在读取数据: {os.path.basename(data_path)}...")
    df = pd.read_csv(data_path)

    # 剔除目标列，仅保留特征列
    # 注意：这里的特征顺序必须与训练模型时完全一致
    target_col = 'conductivity_class'
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df

    # 确保只使用数值类型 (防止意外混入字符串)
    X = X.select_dtypes(include=[np.number])

    print(f"      数据形状: {X.shape}")
    print(f"[3/4] 正在计算 SHAP 值 (可能需要几分钟)...")

    # 使用 TreeExplainer 进行解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --- 处理多分类模型 ---
    # LightGBM 多分类的 shap_values 是一个 list，包含 [class_0_shap, class_1_shap, class_2_shap]
    # 我们通常需要分析特定的某一个类别，或者取平均。
    # 这里我们将保存完整数据，留给绘图代码决定画哪个类。

    # 1. 保存用于绘图的完整数据包 (.joblib 格式最稳定)
    plot_data = {
        "X": X,  # 特征矩阵
        "shap_values": shap_values,  # SHAP值矩阵
        "feature_names": X.columns.tolist()
    }

    joblib_file = os.path.join(output_dir, 'shap_summary_plot_data.joblib')
    joblib.dump(plot_data, joblib_file)

    # 2. 生成一个可读的 CSV 排名文件 (取所有类别的平均绝对值)
    if isinstance(shap_values, list):
        # 对多分类，先取绝对值，再对类别求平均，再对样本求平均
        mean_abs_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    ranking_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': mean_abs_shap
    }).sort_values(by='Importance', ascending=False)

    csv_file = os.path.join(output_dir, 'feature_importance_ranking.csv')
    ranking_df.to_csv(csv_file, index=False)

    print(f"[4/4] 分析完成！")
    print(f"      > 绘图数据已保存至: {joblib_file} (请用绘图代码读取此文件)")
    print(f"      > 特征排名已保存至: {csv_file}")


if __name__ == "__main__":
    # 请根据你的实际路径修改这里
    MODEL_PATH = r"models/lgb_model_20251028_110128.joblib"
    # 使用增强后的数据集，因为它是模型训练的直接依据
    DATA_PATH = r"set/augmented_work_set_20251028_123020.csv"

    run_feature_analysis(MODEL_PATH, DATA_PATH)