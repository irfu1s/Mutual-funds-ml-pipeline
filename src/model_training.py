### LOGIC:
# 1. Load engineered features from data/sip_features.csv
# 2. Drop rows with missing values (bad or incomplete schemes)
# 3. Set target variable → cagr_percent (what we want to predict)
# 4. Build feature matrix X by dropping ID + target + non-feature columns
# 5. Scale features using StandardScaler (helps XGBoost handle ranges)
# 6. Train an XGBRegressor model on all data (baseline full-data fit)
# 7. Save:
#       - Trained model → models/final_xgb_model.json
#       - Scaler → models/scaler.pkl
#       - Feature column names → models/feature_columns.json

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

FEATURE_PATH = Path("data/sip_features_model_ready.csv")
MODEL_PATH = Path("models/final_xgb_model.json")
SCALER_PATH = Path("models/scaler.pkl")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")


def load_features():
    df = pd.read_csv(FEATURE_PATH)

    # Drop any rows with missing key values
    df = df.dropna()
    print(f"Loaded features: {df.shape}")

    return df


def prepare_data(df: pd.DataFrame):
    # Target: what we want the model to learn
    y = df["cagr_percent"].astype(float)

    # Drop identifier, target and date-like columns from features
    drop_cols = [
        "scheme_code",
        "scheme_name",
        "cagr_percent",
        "last_nav_date",   # if present
    ]

    X = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Keep ONLY numeric columns as features
    X = X.select_dtypes(include=[np.number])

    # 🔥 Critical cleanup step: remove inf / -inf and any resulting NaNs
    # Replace infinities with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop rows where ANY feature is NaN after that
    mask_finite = ~X.isna().any(axis=1)
    X = X[mask_finite]
    y = y[mask_finite]

    print(f"Using {len(X.columns)} features after numeric filter.")
    print(f"Training rows after dropping inf/NaN: {len(X)}")

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns.tolist(), scaler


def train_model(X, y):
    # XGBoost regressor as the "brain" of the system
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(X, y)
    return model


def save_artifacts(model, scaler, feature_cols):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save_model(MODEL_PATH)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)

    # Save feature column names as JSON
    import json
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_cols, f)

    print("✅ Saved model, scaler, and feature metadata.")


def main():
    df = load_features()
    X, y, feature_cols, scaler = prepare_data(df)
    model = train_model(X, y)
    save_artifacts(model, scaler, feature_cols)

    print("\n🎯 MODEL TRAINING COMPLETE")
    print(f"Model saved at:   {MODEL_PATH}")
    print(f"Scaler saved at:  {SCALER_PATH}")
    print(f"Features saved at:{FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()