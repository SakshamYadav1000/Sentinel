import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt


def build_model(model_name: str, imbalance: str):
    # Classifier
    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, n_jobs=None)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Imbalance strategy
    sampler = None
    if imbalance == "smote":
        sampler = SMOTE(random_state=42)
    elif imbalance == "undersample":
        sampler = RandomUnderSampler(random_state=42)
    elif imbalance == "none":
        sampler = None
    else:
        raise ValueError(f"Unknown imbalance option: {imbalance}")

    steps = []
    # Scale numeric features (helps distance-based SMOTE and LR)
    steps.append(("scaler", StandardScaler()))
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", clf))

    pipe = ImbPipeline(steps=steps)
    return pipe


def plot_and_save_curves(y_true, y_score, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    roc_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    pr_path = os.path.join(out_dir, "pr_curve.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()

    return {"roc_curve_path": roc_path, "pr_curve_path": pr_path}


def main():
    parser = argparse.ArgumentParser(description="Train a credit card fraud detection model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to creditcard.csv")
    parser.add_argument("--target_column", type=str, default="Class", help="Name of target column")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="logistic_regression", choices=["logistic_regression", "random_forest"])
    parser.add_argument("--imbalance", type=str, default="smote", choices=["smote", "undersample", "none"])
    parser.add_argument("--out_models", type=str, default="models", help="Directory to save models")
    parser.add_argument("--out_reports", type=str, default="reports", help="Directory to save reports")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data_path)
    if args.target_column not in df.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in dataset. Found: {list(df.columns)[:10]}...")

    # Separate features/target
    y = df[args.target_column].astype(int).values
    X = df.drop(columns=[args.target_column])

    # Basic numeric-only safeguard (Kaggle data is numeric already)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Build and fit pipeline
    model = build_model(args.model, args.imbalance)
    model.fit(X_train, y_train)

    # Predict + scores
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        # fall back to decision_function if available
        if hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            # If no score method, use predictions as scores (not ideal)
            y_scores = model.predict(X_test)

    y_pred = (y_scores >= 0.5).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    os.makedirs(args.out_reports, exist_ok=True)
    os.makedirs(args.out_models, exist_ok=True)

    # Save metrics
    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    with open(os.path.join(args.out_reports, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save curves
    curve_paths = plot_and_save_curves(y_test, y_scores, args.out_reports)

    # Save model
    model_path = os.path.join(args.out_models, "model.joblib")
    joblib.dump(model, model_path)

    # Human-readable summary
    print("=== Training complete ===")
    print(f"Model: {args.model}, Imbalance: {args.imbalance}")
    print(f"ROC-AUC: {roc_auc:.5f}")
    print(f"PR-AUC:  {pr_auc:.5f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {os.path.join(args.out_reports, 'metrics.json')}")
    print(f"Saved plots to: {curve_paths['roc_curve_path']} and {curve_paths['pr_curve_path']}")


    import joblib

    # Save model (consistent naming)
    model_path = os.path.join(args.out_models, "fraud_detector.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")


 # Save human-readable report
    report_txt_path = os.path.join(args.out_reports, "metrics.txt")
    with open(report_txt_path, "w") as f:
        f.write("=== Model Evaluation Report ===\n")
        f.write(f"Model: {args.model}, Imbalance: {args.imbalance}\n\n")
        f.write(f"ROC-AUC: {roc_auc:.5f}\n")
        f.write(f"PR-AUC:  {pr_auc:.5f}\n\n")
        f.write("Confusion Matrix [[TN, FP], [FN, TP]]:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
    print(f"Saved human-readable report to: {report_txt_path}")

if __name__ == "__main__":
    main()
