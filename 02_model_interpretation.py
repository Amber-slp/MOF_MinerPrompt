import pandas as pd
import numpy as np
import shap
import joblib
import os


def analyze_trained_model(model_path, data_path, output_dir='./results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    target_col = 'conductivity_class'
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df
    X = X.select_dtypes(include=[np.number])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    analysis_pack = {
        "X": X,
        "shap_values_list": shap_values, 
        "feature_names": X.columns.tolist()
    }

    output_file = os.path.join(output_dir, 'model_shap_data.joblib')
    joblib.dump(analysis_pack, output_file)



if __name__ == "__main__":
    MODEL_PATH = r"models/lgb_model_20251028_110128.joblib"
    DATA_PATH = r"set/augmented_work_set_20251028_123020.csv"

    analyze_trained_model(MODEL_PATH, DATA_PATH)
