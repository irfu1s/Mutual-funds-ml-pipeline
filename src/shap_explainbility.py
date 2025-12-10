### LOGIC:
# 1. Load trained model + scaler + feature columns
# 2. Load sip_features.csv (the dataset)
# 3. Scale features exactly like during training
# 4. Compute SHAP values using TreeExplainer
# 5. Save:
#       - summary plot (global)
#       - dependence plots for key features
#       - optional force plot for a specific scheme
# 6. These plots help us understand:
#       - which features matter most
#       - how the model behaves
#       - why recommendations are made


import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import json

FEATURE_PATH = Path("data/sip_features_model_ready.csv")
MODEL_PATH = Path("models/final_xgb_model.json")
SCALER_PATH = Path("models/scaler.pkl")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")
OUT_DIR = Path("shap_exports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_cols = json.load(f)

    return model, scaler, feature_cols


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and artifacts
    df = pd.read_csv(FEATURE_PATH).dropna()
    model, scaler, feature_cols = load_artifacts()

    # Prepare features
    X = df[feature_cols]
    X_scaled = scaler.transform(X)

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # 1. Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Dependence plots for the top 3 features
    top_features = feature_cols[:3]
    for feat in top_features:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X, show=False)
        plt.savefig(OUT_DIR / f"dependence_{feat}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("SHAP explainability files created in shap_exports/")


if __name__ == "__main__":
    main()