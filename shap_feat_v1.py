import pandas as pd
import numpy as np
import shap
import joblib
import os


def run_feature_analysis(model_path, data_path, output_dir='./results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[1/4] {os.path.basename(model_path)}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"failure: {e}")
        return

    print(f"[2/4] read: {os.path.basename(data_path)}...")
    df = pd.read_csv(data_path)

    target_col = 'conductivity_class'
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df

    X = X.select_dtypes(include=[np.number])

    print(f"      shape: {X.shape}")
    print(f"[3/4] ...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plot_data = {
        "X": X, 
        "shap_values": shap_values,
        "feature_names": X.columns.tolist()
    }

    joblib_file = os.path.join(output_dir, 'shap_summary_plot_data.joblib')
    joblib.dump(plot_data, joblib_file)

    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    ranking_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': mean_abs_shap
    }).sort_values(by='Importance', ascending=False)

    csv_file = os.path.join(output_dir, 'feature_importance_ranking.csv')
    ranking_df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    MODEL_PATH = r"models/lgb_model_20251028_110128.joblib"
    DATA_PATH = r"set/augmented_work_set_20251028_123020.csv"

    run_feature_analysis(MODEL_PATH, DATA_PATH)
