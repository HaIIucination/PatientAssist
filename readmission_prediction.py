# problem1_readmission_prediction.py
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from xgboost import XGBClassifier
import shap

RANDOM_STATE = 42


# -------------------------
# 1) Load local dataset
# -------------------------
def load_local_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError("Unsupported file type. Please provide a .csv or .parquet file.")


# -------------------------
# 2) Robust target parsing (y must be 0/1 int)
# -------------------------
def coerce_binary_target(y: pd.Series) -> pd.Series:
    """
    Coerce a target series into 0/1 integer labels.
    Supports:
      - int/float already 0/1
      - boolean True/False
      - strings: 'yes'/'no', 'true'/'false', '1'/'0', '<30'/'NO'/' >30 ' (common UCI legacy)
    """
    # If it's already numeric-ish
    if pd.api.types.is_numeric_dtype(y):
        y2 = pd.to_numeric(y, errors="coerce")
        # If values aren't just 0/1, this might be a wrong target column
        # but we still try to binarize if it's close
        uniq = set(pd.Series(y2.dropna().unique()).astype(float).tolist())
        if uniq.issubset({0.0, 1.0}):
            return y2.astype(int)
        # Otherwise, raise so you don't silently do the wrong thing
        raise ValueError(f"Target appears numeric but not binary 0/1. Unique values: {sorted(list(uniq))[:20]}")

    # Boolean
    if pd.api.types.is_bool_dtype(y):
        return y.astype(int)

    # Object/string handling
    y_str = y.astype(str).str.strip().str.lower()

    # Common mappings
    mapping = {
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
        "t": 1, "f": 0,
        # UCI legacy readmission values:
        "<30": 1,
        ">30": 0,
        "no": 0
    }

    # If all values map, do it
    uniq = set(y_str.unique())
    if uniq.issubset(set(mapping.keys())):
        return y_str.map(mapping).astype(int)

    # Sometimes there are NaNs or unexpected tokens; try mapping and validate
    y_mapped = y_str.map(mapping)
    if y_mapped.notna().mean() > 0.95:
        # If almost everything mapped, assume the rest are missing/weird
        return y_mapped.fillna(0).astype(int)

    raise ValueError(
        "Could not coerce target to binary 0/1. "
        f"Sample unique values: {list(y_str.unique())[:20]}"
    )


# -------------------------
# 3) Choose target & prevent leakage by dropping other label columns
# -------------------------
def split_X_y(df: pd.DataFrame, target: str | None = None, label_cols: list[str] | None = None):
    """
    Returns:
      X: feature dataframe
      y: binary target series (0/1)
      target_col: name of chosen target

    Behavior:
    - If target is provided, it is used.
    - Else auto-detects using a priority list.
    - Drops ALL other label-like columns from X to prevent leakage.
    """
    df = df.copy()

    # Default label columns you might have locally after preprocessing
    default_label_cols = ["readmit_30_days", "readmit_binary", "readmitted_30d", "readmitted"]
    if label_cols is None:
        label_cols = default_label_cols
    else:
        # keep user-provided plus defaults (avoid missing common ones)
        label_cols = list(dict.fromkeys(label_cols + default_label_cols))

    # Auto-detect target if not provided
    if target is None:
        # Prefer the most specific first
        candidates = ["readmit_30_days", "readmit_binary", "readmitted_30d", "readmitted"]
        target = next((c for c in candidates if c in df.columns), None)

    if target is None or target not in df.columns:
        raise ValueError(
            f"Target column not found. Provide --target. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Build y from selected target column values
    y = coerce_binary_target(df[target])

    # Drop other label columns from X to prevent leakage
    cols_to_drop = [c for c in label_cols if c in df.columns and c != target]
    X = df.drop(columns=[target] + cols_to_drop, errors="ignore")

    print(f"[INFO] Using target column: {target}")
    if cols_to_drop:
        print(f"[INFO] Dropping potential label leakage columns from features: {cols_to_drop}")

    return X, y, target


# -------------------------
# 4) Optional: drop ID-like columns
# -------------------------
def drop_id_columns(X: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in id_cols:
        if c in X.columns:
            X = X.drop(columns=[c])
    return X


# -------------------------
# 5) Feature engineering (Fairlearn-style safe)
# -------------------------
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Safe feature engineering for Fairlearn-style columns:
    - A1Cresult -> a1c_severity ordinal
    - max_glu_serum -> glu_severity ordinal
    - utilization score if had_emergency/had_inpatient_days/had_outpatient_days exist
    - meds_x_los interaction if num_medications & time_in_hospital exist
    """
    X = X.copy()

    a1c_map = {"none": 0, "normal": 1, ">7": 2, ">8": 3}
    glu_map = {"none": 0, "norm": 1, ">200": 2, ">300": 3}

    if "A1Cresult" in X.columns:
        X["a1c_severity"] = (
            X["A1Cresult"].astype(str).str.strip().str.lower().map(a1c_map)
        )

    if "max_glu_serum" in X.columns:
        X["glu_severity"] = (
            X["max_glu_serum"].astype(str).str.strip().str.lower().map(glu_map)
        )

    util_cols = [c for c in ["had_emergency", "had_inpatient_days", "had_outpatient_days"] if c in X.columns]
    if util_cols:
        for c in util_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X["util_score"] = X[util_cols].sum(axis=1)

    if "num_medications" in X.columns and "time_in_hospital" in X.columns:
        X["meds_x_los"] = pd.to_numeric(X["num_medications"], errors="coerce") * pd.to_numeric(X["time_in_hospital"], errors="coerce")

    return X


# -------------------------
# 6) Preprocessor builder
# -------------------------
def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


# -------------------------
# 7) Model builder (imbalance aware)
# -------------------------
def build_model(scale_pos_weight: float):
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# -------------------------
# 8) Plot curves
# -------------------------
def plot_curves(y_true, y_proba, out_dir: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.tight_layout()
    path = os.path.join(out_dir, "roc_pr_curves.png")
    plt.savefig(path, dpi=200)
    plt.show()
    print(f"[INFO] Saved ROC/PR curves to: {path}")


# -------------------------
# 9) Train / eval with proper split
# -------------------------
def train_evaluate(X: pd.DataFrame, y: pd.Series, out_dir: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    preprocessor = build_preprocessor(X_train)
    model = build_model(scale_pos_weight)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Validation
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_auroc = roc_auc_score(y_val, val_proba)
    val_auprc = average_precision_score(y_val, val_proba)
    print("\n--- Validation ---")
    print(f"AUROC: {val_auroc:.4f}")
    print(f"AUPRC: {val_auprc:.4f}")

    # Test
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    auroc = roc_auc_score(y_test, test_proba)
    auprc = average_precision_score(y_test, test_proba)

    print("\n--- Test ---")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, test_pred))

    plot_curves(y_test, test_proba, out_dir)

    metrics = {
        "val_auroc": float(val_auroc),
        "val_auprc": float(val_auprc),
        "test_auroc": float(auroc),
        "test_auprc": float(auprc),
        "pos_rate": float(y.mean()),
        "train_pos": pos,
        "train_neg": neg,
        "scale_pos_weight": float(scale_pos_weight),
    }

    return pipeline, X_test, y_test, metrics


# -------------------------
# 10) SHAP explainability
# -------------------------
def get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame):
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        n = preprocessor.transform(X.head(1)).shape[1]
        return np.array([f"f_{i}" for i in range(n)])


def shap_global_summary(pipeline: Pipeline, X_sample: pd.DataFrame, out_dir: str, max_display=20):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    Xt = preprocessor.transform(X_sample)
    feat_names = get_feature_names(preprocessor, X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xt)

    shap.summary_plot(
        shap_values,
        features=Xt,
        feature_names=feat_names,
        max_display=max_display,
        show=False
    )
    path = os.path.join(out_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()
    print(f"[INFO] Saved SHAP summary plot to: {path}")


def explain_one_patient(pipeline: Pipeline, X_one: pd.DataFrame, top_k=10):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    Xt = preprocessor.transform(X_one)
    feat_names = get_feature_names(preprocessor, X_one)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xt)[0]
    proba = pipeline.predict_proba(X_one)[:, 1][0]

    idx = np.argsort(np.abs(sv))[::-1][:top_k]
    reasons = [(str(feat_names[i]), float(sv[i])) for i in idx]
    return float(proba), reasons


# -------------------------
# 11) Save artifacts
# -------------------------
def save_artifacts(pipeline: Pipeline, metrics: dict, out_dir: str):
    model_path = os.path.join(out_dir, "readmission_pipeline.pkl")
    metrics_path = os.path.join(out_dir, "metrics.json")

    joblib.dump(pipeline, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved metrics to: {metrics_path}")


# -------------------------
# Main entry
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Readmission prediction (local preprocessed dataset).")
    parser.add_argument("--data", required=True, help="Path to preprocessed dataset (.csv or .parquet)")
    parser.add_argument("--target", default=None, help="Target column name (recommended to pass explicitly)")
    parser.add_argument(
        "--label-cols",
        default="readmit_30_days,readmit_binary,readmitted_30d,readmitted",
        help="Comma-separated list of label-like columns to drop from features (except chosen target)."
    )
    parser.add_argument("--outdir", default="artifacts_problem1", help="Output directory for artifacts")
    parser.add_argument("--drop-cols", default="", help="Comma-separated columns to drop (IDs etc.)")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP explainability plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_local_data(args.data)

    # Parse leakage label columns list
    label_cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]

    # Split into features/target and prevent leakage by dropping other label columns
    X, y, target_col = split_X_y(df, target=args.target, label_cols=label_cols)

    # Drop custom columns if requested (IDs etc.)
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    X = drop_id_columns(X, drop_cols)

    # Feature engineering
    X = engineer_features(X)

    print(f"\nLoaded dataset: {args.data}")
    print(f"Rows/Cols: {df.shape}")
    print(f"Target used: {target_col}")
    print(f"Positive rate (readmitted within 30d): {y.mean():.4f}")

    pipeline, X_test, y_test, metrics = train_evaluate(X, y, args.outdir)
    save_artifacts(pipeline, metrics, args.outdir)

    # Explainability
    if not args.no_shap:
        sample = X_test.sample(min(300, len(X_test)), random_state=RANDOM_STATE)
        shap_global_summary(pipeline, sample, args.outdir, max_display=20)

        one = X_test.sample(1, random_state=RANDOM_STATE)
        proba, reasons = explain_one_patient(pipeline, one, top_k=10)
        print("\n--- Example patient explanation ---")
        print(f"Predicted readmission risk probability: {proba:.3f}")
        print("Top contributing factors (feature, SHAP contribution):")
        for fname, contrib in reasons:
            print(f"  - {fname}: {contrib:+.4f}")
    else:
        print("\n[INFO] SHAP disabled (--no-shap).")


if __name__ == "__main__":
    main()

