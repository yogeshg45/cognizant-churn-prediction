from flask import (
    Flask, render_template, request, send_from_directory,
    redirect, url_for, flash, jsonify
)
import joblib
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename
import gdown
import google.generativeai as genai


# Gemini LLM API key and setup
GEMINI_API_KEY = "AIzaSyDh_q12etYVVvBmqqqZzfO5aGiWZ2Z-lB4"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# Google Drive Model IDs and paths
DRIVE_MODEL_ID = "1MihDLFj1ZRnVG8j5UB2KR9h71xZwsU-M"
MODEL_FILENAME = "churn_pipeline.joblib"
MODEL_PATH = MODEL_FILENAME
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_MODEL_ID}"


def download_model_if_missing(model_path=MODEL_PATH):
    """Download model from Google Drive if not already present."""
    if not os.path.exists(model_path):
        print(f"[INFO] Downloading model from Google Drive: {DRIVE_URL}")
        gdown.download(DRIVE_URL, model_path, quiet=False)
    else:
        print("[INFO] Model file exists locally.")


def load_model(path=MODEL_PATH):
    """
    Load model from joblib file.
    If a dict is saved, return first valid model with predict method.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    obj = joblib.load(path)
    if hasattr(obj, "predict"):
        return obj
    if isinstance(obj, dict):
        for key, val in obj.items():
            if hasattr(val, "predict"):
                print(f"[INFO] Using model inside dict at key '{key}'")
                return val
        raise ValueError("No model with .predict found inside joblib dict.")
    raise TypeError("Unsupported model object stored in joblib file.")


# Flask app config
app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-random-secret-key"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def select_feature_columns(df):
    """
    Return a list of feature columns to use for prediction.
    Prefer feature_0..feature_15, else numeric excluding blacklist.
    """
    pref = [f"feature_{i}" for i in range(16)]
    if all(col in df.columns for col in pref):
        return pref
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    blacklist = {"Churn", "Prediction", "Churn_Probability", "Actual"}
    return [col for col in numeric_cols if col not in blacklist]


def get_recommendation(prob):
    """Return business recommendation based on churn probability."""
    if prob is None:
        return "No probability available."
    if prob >= 0.8:
        return "🚨 Very high risk: Immediate personalized retention offers, loyalty rewards, and direct outreach."
    elif prob >= 0.6:
        return "⚠️ High risk: Prioritize onboarding improvements, targeted campaigns, and proactive support."
    elif prob >= 0.4:
        return "🔎 Medium risk: Monitor behavior, provide helpful tips, and engage with value-added services."
    else:
        return "✅ Low risk: Maintain engagement with regular communication and satisfaction surveys."


def get_aggregate_recommendation(avg_prob):
    """Company-wide recommendations based on average churn probability."""
    if avg_prob is None:
        return "No predictions available yet."
    if avg_prob >= 0.6:
        return "⚠️ Overall churn risk is high. Focus on claims processing speed, loyalty rewards, and proactive retention strategies."
    elif avg_prob >= 0.4:
        return "🔎 Moderate churn risk. Segment customers and strengthen onboarding."
    else:
        return "✅ Overall churn risk is low. Keep monitoring and maintain strong engagement."


# Global variable to hold loaded model and latest predictions path
model = None
latest_predictions_path = None


# Load model at startup
try:
    download_model_if_missing()
    model = load_model()
    print("[INFO] Model loaded successfully.")
except Exception as err:
    print(f"[WARN] Could not load model: {err}")
    model = None


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_predict():
    global latest_predictions_path, model
    if request.method == "POST":
        if model is None:
            flash("Prediction model not loaded on server. Please contact admin.", "error")
            return redirect(request.url)
        if "file" not in request.files:
            flash("No file part in request.", "error")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected for upload.", "error")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath)
            except Exception:
                flash("Failed to read CSV file. Please check formatting.", "error")
                return redirect(request.url)
            feat_cols = select_feature_columns(df)
            if len(feat_cols) == 0:
                flash("No valid numeric features found for prediction.", "error")
                return redirect(request.url)
            X = df[feat_cols]
            try:
                preds = model.predict(X)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[:, 1]
                else:
                    probs = [None] * len(preds)
            except Exception as e:
                flash(f"Prediction error: {e}", "error")
                return redirect(request.url)
            df["Prediction"] = ["Churn" if int(p) == 1 else "No Churn" for p in preds]
            df["Churn_Probability"] = [round(float(p), 3) if p is not None else None for p in probs]
            df["Recommendation"] = [get_recommendation(p) for p in df["Churn_Probability"]]
            out_name = f"predictions_{filename}"
            out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
            df.to_csv(out_path, index=False)
            latest_predictions_path = out_path
            flash(f"Predictions completed for {len(df)} records.", "success")
            return redirect(url_for("results", filename=out_name))
        else:
            flash("Invalid file format. Only CSV allowed.", "error")
            return redirect(request.url)
    return render_template("upload.html")


@app.route("/results/<path:filename>")
def results(filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(path):
        flash("Results file not found.", "error")
        return redirect(url_for("upload_predict"))
    df = pd.read_csv(path)
    preview_html = df.head(10).to_html(classes="table table-striped", index=False)
    total = len(df)
    churned = int((df["Prediction"] == "Churn").sum()) if "Prediction" in df.columns else 0
    avg_prob = float(df["Churn_Probability"].mean()) if "Churn_Probability" in df.columns else None
    agg_recommendation = get_aggregate_recommendation(avg_prob)
    return render_template("results.html",
                           preview_table=preview_html,
                           download_link=url_for("download_file", filename=filename),
                           total=total, churned=churned, avg_prob=avg_prob,
                           agg_recommendation=agg_recommendation)


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)


@app.route("/predict_single", methods=["POST"])
def predict_single():
    global model
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500
    try:
        feat_cols = [f"feature_{i}" for i in range(16)]
        data = {}
        provided = False
        for c in feat_cols:
            v = request.form.get(c)
            if v and v.strip() != "":
                data[c] = float(v)
                provided = True
        if not provided:
            # fallback: accept any numeric input fields
            for k, v in request.form.items():
                try:
                    data[k] = float(v)
                except:
                    pass
        if not data:
            return jsonify({"error": "No numeric input provided."}), 400
        X = pd.DataFrame([data])
        cols = select_feature_columns(X)
        X = X[cols] if cols else X
        pred = model.predict(X)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = float(model.predict_proba(X)[:, 1][0])
            except Exception:
                prob = None
        return jsonify({
            "prediction": "Churn" if int(pred) == 1 else "No Churn",
            "probability": round(prob, 3) if prob is not None else None,
            "recommendation": get_recommendation(prob)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard")
def dashboard():
    global latest_predictions_path, model
    if not latest_predictions_path or not os.path.exists(latest_predictions_path):
        flash("No predictions available. Upload a file first.", "info")
        return redirect(url_for("upload_predict"))
    df = pd.read_csv(latest_predictions_path)
    total = len(df)
    churn_count = int((df["Prediction"] == "Churn").sum()) if "Prediction" in df.columns else 0
    no_churn_count = total - churn_count
    avg_prob = round(float(df["Churn_Probability"].mean()), 3) if "Churn_Probability" in df.columns else None
    pie_labels = ["Churn", "No Churn"]
    pie_values = [churn_count, no_churn_count]
    probs = df["Churn_Probability"].dropna().astype(float).tolist() if "Churn_Probability" in df.columns else []
    feature_importance = []
    try:
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
            sample_cols = select_feature_columns(df)
            names = sample_cols if len(sample_cols) == len(imps) else [f"f{i}" for i in range(len(imps))]
            feature_importance = sorted(zip(names, imps), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        feature_importance = []
    return render_template("dashboard.html",
                           total=total,
                           churn_count=churn_count,
                           no_churn_count=no_churn_count,
                           avg_prob=avg_prob,
                           pie_labels=json.dumps(pie_labels),
                           pie_values=json.dumps(pie_values),
                           probs=json.dumps(probs),
                           feature_importance=feature_importance)


@app.route("/recommendations")
def recommendations():
    global latest_predictions_path
    avg_prob = None
    agg_recommendation = None
    if latest_predictions_path and os.path.exists(latest_predictions_path):
        df = pd.read_csv(latest_predictions_path)
        if "Churn_Probability" in df.columns:
            avg_prob = float(df["Churn_Probability"].mean())
            agg_recommendation = get_aggregate_recommendation(avg_prob)
    return render_template("recommendations.html",
                           avg_prob=avg_prob,
                           agg_recommendation=agg_recommendation)


@app.route("/about")
def about():
    text = (
        "Insurance companies operate in a highly competitive environment. "
        "With data from millions of customers, understanding churn is challenging. "
        "This tool helps predict churn using 16 anonymized features (feature_0..feature_15)."
    )
    return render_template("about.html", project_text=text)


@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


@app.route("/api/chatbot", methods=["POST"])
def chatbot_api():
    user_message = request.json.get("message", "")
    churn_model_outputs = ""
    if latest_predictions_path and os.path.exists(latest_predictions_path):
        df = pd.read_csv(latest_predictions_path)
        try:
            top_churn = df.sort_values("Churn_Probability", ascending=False).iloc[0]
            churn_model_outputs = (
                f"Sample churn prediction:\n"
                f"Churn Probability: {top_churn['Churn_Probability']}\n"
                f"Recommendation: {top_churn['Recommendation']}\n"
            )
        except Exception:
            churn_model_outputs = ""
    context_prompt = (
        "You are an AI assistant for insurance churn analysis. "
        "You have access to the latest churn prediction results. "
        "When the user asks about churn, customer behavior, or prevention strategies, "
        "use the following prediction insights to respond:\n"
        f"{churn_model_outputs}\n"
        "Answer in a clear, business-friendly style with next steps."
    )
    full_prompt = context_prompt + "\nUser: " + user_message
    try:
        response = gemini_model.generate_content(full_prompt)
        reply = response.text
    except Exception as e:
        reply = f"Sorry, AI service is unavailable. ({e})"
    return jsonify({"reply": reply})


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
