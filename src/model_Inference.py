### LOGIC:
# 1. Load:
#       - trained model (final_xgb_model.json)
#       - scaler (scaler.pkl)
#       - feature column list (feature_columns.json)
# 2. Load engineered features from data/sip_features.csv
# 3. Build feature matrix X using the same columns used during training
# 4. Scale X using the loaded scaler
# 5. Use the model to predict cagr_percent for each scheme
# 6. Attach predictions back to:
#       - scheme_code
#       - scheme_name
# 7. Return a DataFrame with:
#       - scheme_code
#       - scheme_name
#       - predicted_cagr_percent
#       - (optionally keep other feature columns for downstream logic)
# 8. Optionally save predictions to data/sip_predictions.csv
# 9. This module will be used by:
#       - recommender system (to choose top schemes)
#       - NLP agent (to answer user questions)
#       - app UI (to display results)


import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
import joblib
import json

FEATURE_PATH = Path("data/sip_features_model_ready.csv")
MODEL_PATH = Path("models/final_xgb_model.json")
SCALER_PATH = Path("models/scaler.pkl")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")
PREDICTIONS_PATH = Path("data/sip_predictions.csv")


def load_artifacts():
    """Load trained model, scaler, and feature column names."""
    # Load model
    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Load feature column list
    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_cols = json.load(f)

    return model, scaler, feature_cols


def load_feature_data():
    """Load engineered feature dataset."""
    df = pd.read_csv(FEATURE_PATH)

    # Drop rows with missing values (safety)
    df = df.dropna()

    # Ensure scheme_code as string (for consistency)
    if "scheme_code" in df.columns:
        df["scheme_code"] = df["scheme_code"].astype(str)

    return df


def predict_for_dataframe(df, model, scaler, feature_cols):
    """
    Given a DataFrame with feature columns,
    return predictions with scheme identifiers attached.
    """

    # Build feature matrix
    X = df[feature_cols]

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    preds = model.predict(X_scaled)

    # Build output
    out_cols = []
    if "scheme_code" in df.columns:
        out_cols.append("scheme_code")
    if "scheme_name" in df.columns:
        out_cols.append("scheme_name")

    result = df[out_cols].copy()
    result["predicted_cagr_percent"] = preds

    return result


def predict_all_schemes(save_to_csv: bool = True):
    """
    High-level function:
    - loads data + model
    - predicts for all schemes in sip_features.csv
    - optionally saves predictions to CSV
    - returns DataFrame with predictions
    """
    df = load_feature_data()
    model, scaler, feature_cols = load_artifacts()

    predictions = predict_for_dataframe(df, model, scaler, feature_cols)

    if save_to_csv:
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(PREDICTIONS_PATH, index=False)
        print(f"✅ Saved predictions → {PREDICTIONS_PATH}")

    print("Sample predictions:")
    print(predictions.head())

    return predictions


def main():
    predict_all_schemes(save_to_csv=True)


if __name__ == "__main__":
    main()