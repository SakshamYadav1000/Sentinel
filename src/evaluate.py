# src/evaluate.py
import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

def evaluate(model_path, data_path, report_dir="reports"):
    # Make reports directory if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)

    # Load model & data
    model = joblib.load(model_path)
    X, y = load_data(data_path)

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)

    # Text report
    report_text = []
    report_text.append("=== Evaluation Report ===")
    report_text.append(f"ROC-AUC: {roc_auc:.4f}")
    report_text.append(f"PR-AUC : {pr_auc:.4f}")
    report_text.append("\nClassification Report:\n")
    report_text.append(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    report_text.append("Confusion Matrix:\n" + str(cm))

    print("\n".join(report_text))

    # Save text report
    report_path = os.path.join(report_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_text))
    print(f"\n✅ Evaluation report saved at: {report_path}")

    # Save confusion matrix as CSV
    cm_path = os.path.join(report_dir, "confusion_matrix.csv")
    pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_csv(cm_path)
    print(f"✅ Confusion matrix saved at: {cm_path}")

    # ROC & PR curves
    fpr, tpr, _ = roc_curve(y, y_prob)
    precision, recall, _ = precision_recall_curve(y, y_prob)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.tight_layout()

    # Save plots
    roc_pr_path = os.path.join(report_dir, "roc_pr_curves.png")
    plt.savefig(roc_pr_path)
    print(f"✅ ROC & PR curves saved at: {roc_pr_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model .pkl")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset")
    args = parser.parse_args()

    evaluate(args.model_path, args.data_path)