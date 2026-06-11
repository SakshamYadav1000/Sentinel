from flask import Flask, request, render_template, redirect, url_for, session, send_file
import joblib
import pandas as pd
import os
import numpy as np

#path setup:
project_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#template path
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
#result path:
results=os.path.join(project_root,"results")
os.makedirs(results,exist_ok=True)

# Flask setup
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = Flask(
    __name__,
    template_folder=template_dir,
    static_folder=static_dir
)

app.secret_key = os.environ.get("SECRET_KEY", "sentinel-dev-key")

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detector.pkl")

model = joblib.load(MODEL_PATH)


# Simple authentication
users = {"admin": "password123"}

@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    if username in users and users[username] == password:
        session["username"] = username
        return redirect(url_for("dashboard"))
    return "Invalid credentials", 401

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))

@app.route("/guest")
def guest():
    session["username"] = "Guest"
    return redirect(url_for("dashboard"))

#Dashboard:
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "username" not in session:
        return redirect(url_for("home"))

    result = None
    return render_template("dashboard.html", user=session["username"], result=result)

#csv upload:
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "username" not in session:
        return redirect(url_for("home"))

    file = request.files.get("file")
    if not file or file.filename == "":
        return "No file uploaded", 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Error reading CSV: {e}", 400

    expected_features = 30  # change according to your model
    if df.shape[1] != expected_features:
        return f"CSV must have {expected_features} columns, got {df.shape[1]}", 400

    # Predict
    X = df.values.astype(float)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X)[:, 1]
    else:
        y_scores = model.predict(X)
    y_pred = (y_scores >= 0.5).astype(int)

    df["prediction"] = y_pred
    df["fraud_probability"] = y_scores

    # Save full result for download
    os.makedirs("results", exist_ok=True)
    result_csv_path = os.path.join("results", "fraud_results.csv")
    df.to_csv(result_csv_path, index=False)

    # Prepare result summary for template (top 10 frauds only)
    fraud_df = df[df["prediction"] == 1].head(10)
    result = {
        "total": len(df),
        "frauds": int(sum(y_pred)),
        "columns": fraud_df.columns.tolist() if not fraud_df.empty else [],
        "top10": fraud_df.values.tolist() if not fraud_df.empty else []
    }

    return render_template("dashboard.html", user=session["username"], result=result)

#Saving for only frauds
@app.route("/download_frauds")
def download_frauds():
    if "username" not in session:
        return redirect(url_for("home"))
    
    df=pd.read_csv("results/fraud_results.csv")
    frauds=df[df["prediction"]==1]
    frauds.to_csv("results/frauds.csv",index=False)

    frauds_csv_path = os.path.join(results, "frauds.csv")
    if not os.path.exists(frauds_csv_path):
        return "No result available. Upload a CSV first.", 400
    return send_file(frauds_csv_path, as_attachment=True)

#saving csv in result folder
@app.route("/download_result")
def download_result():
    if "username" not in session:
        return redirect(url_for("home"))
    result_csv_path = os.path.join(results, "fraud_results.csv")
    if not os.path.exists(result_csv_path):
        return "No result available. Upload a CSV first.", 400
    return send_file(result_csv_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)