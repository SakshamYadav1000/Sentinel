# src/tune.py
import argparse
import joblib
import optuna
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

def objective(trial, X, y, cv):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 200)  # smaller range for speed
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42
    )

    scorer = make_scorer(roc_auc_score, needs_proba=True)

    # Progress bar for folds
    scores = []
    for train_idx, test_idx in tqdm(cv.split(X, y), total=cv.get_n_splits(),
                                    desc=f"Trial {trial.number}", leave=False):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict_proba(X.iloc[test_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[test_idx], preds))
    return sum(scores) / len(scores)

def run_tuning(data_path, mode):
    X, y = load_data(data_path)

    if mode == "fast":
        n_trials, n_splits = 5, 2
    else:  # full mode
        n_trials, n_splits = 50, 3

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    study = optuna.create_study(direction="maximize")

    for _ in tqdm(range(n_trials), desc="Overall progress"):
        study.optimize(lambda trial: objective(trial, X, y, cv), n_trials=1)

    print("Best ROC-AUC:", study.best_value)
    print("Best params:", study.best_params)

    # Save best model
    best_model = RandomForestClassifier(
        **study.best_params, n_jobs=-1, random_state=42
    )
    best_model.fit(X, y)
    joblib.dump(best_model, f"models/best_rf_{mode}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV")
    parser.add_argument("--mode", type=str, choices=["fast", "full"], default="fast")
    args = parser.parse_args()

    run_tuning(args.data_path, args.mode)