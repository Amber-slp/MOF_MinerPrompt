import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os


def analyze_raw_data(raw_data_path, output_dir='./results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(raw_data_path)
    
    target_col = 'conductivity_class'

    X = df.drop(columns=[target_col, 'conductivity', 'Smiles'], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df[target_col]

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Raw_Importance': rf.feature_importances_
    }).sort_values(by='Raw_Importance', ascending=False)


    output_file = os.path.join(output_dir, 'raw_data_importance.csv')
    importance_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    RAW_PATH = r"rawdata/raw_v2.csv"
    analyze_raw_data(RAW_PATH)
