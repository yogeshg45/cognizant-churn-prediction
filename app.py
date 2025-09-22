from flask import (
    Flask, render_template, request, send_from_directory,
    redirect, url_for, flash, jsonify
)
import joblib
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename
import zipfile
import google.generativeai as genai

# Gemini API Key - replace with your key in production securely
GEMINI_API_KEY = "AIzaSyDh_q12etYVVvBmqqqZzfO5aGiWZ2Z-lB4"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ZIP = os.path.join(BASE_DIR, "churn_pipeline.zip")
MODEL_FILENAME = "churn_pipeline.joblib"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def unzip_model(zip_path=MODEL_ZIP, extract_to="."):
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"{zip_path} not found. Please upload the zip file or place it in the app directory."
            )
        print(f"[INFO] Extracting model file from {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file {MODEL_FILENAME} not found after extracting zip."
            )

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    data = joblib.load(path)
    if hasattr(data, "predict"):
        return data
    if isinstance(data, dict):
        for key, val in data.items():
            if hasattr(val, "predict"):
                print(f"[INFO] Using model found at dict key: {key}")
                return val
        raise ValueError("No valid model (with .predict) found inside joblib dict.")
    raise ValueError("Unsupported object stored in joblib.")

try:
    unzip_model()
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    model = None
    print(f"[WARN] Model load failed: {e}")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def select_feature_columns(df):
    feat_cols = [f"feature_{i}" for i in range(16)]
    if all(c in df.columns for c in feat_cols):
        return feat_cols
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    blacklist = {"Churn", "Prediction", "Churn_Probability", "Actual"}
    numeric_cols = [c for c in numeric_cols if c not in blacklist]
    return numeric_cols

def get_recommendation(prob):
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
    if avg_prob is None:
        return "No predictions available yet."
    if avg_prob >= 0.6:
        return "⚠️ Overall churn risk is high. Focus on claims processing speed, loyalty rewards, and proactive retention strategies."
    elif avg_prob >= 0.4:
        return "🔎 Moderate churn risk. Segment customers and strengthen onboarding."
    else:
        return "✅ Overall churn risk is low. Keep monitoring and maintain strong engagement."

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace-this-with-a-secure-random-key"
latest_predictions_path = None

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_predict():
    global latest_predictions_path, model
    if request.method == "POST":
        if model is None:
            flash("Prediction model not loaded on server. Please contact admin. (See server logs for more info.)", "error")
            return redirect(request.url)
        if "file" not in request.files:
            flash("No file part", "error")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            feat_cols = select_feature_columns(df)
            if len(feat_cols) == 0:
                flash("No usable numeric features found for prediction (expected feature_0..feature_15 or numeric columns).", "error")
                return redirect(request.url)
            X = df[feat_cols]
            try:
                preds = model.predict(X)
            except Exception as e:
                flash(f"Model prediction error: {e}", "error")
                return redirect(request.url)
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X)[:, 1]
                except Exception:
                    probs = [None] * len(preds)
            else:
                probs = [None] * len(preds)
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
            flash("Invalid file format. Upload a CSV.", "error")
            return redirect(request.url)
    return render_template("upload.html")

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
            if v is not None and v != "":
                data[c] = float(v)
                provided = True
        if not provided:
            for k, v in request.form.items():
                try:
                    data[k] = float(v)
                except:
                    pass
        if len(data) == 0:
            return jsonify({"error": "No numeric input provided."}), 400
        X = pd.DataFrame([data])
        cols = select_feature_columns(X)
        if len(cols) == 0:
            cols = list(X.columns)
        X = X[cols]
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

@app.route("/results/<path:filename>")
def results(filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(path):
        flash("File not found.", "error")
        return redirect(url_for("upload_predict"))
    df = pd.read_csv(path)
    preview_html = df.head(10).to_html(classes="table table-striped", index=False)
    total = len(df)
    churned = int((df["Prediction"] == "Churn").sum()) if "Prediction" in df.columns else 0
    avg_prob = float(df["Churn_Probability"].mean()) if "Churn_Probability" in df.columns else None
    agg_recommendation = get_aggregate_recommendation(avg_prob)
    return render_template(
        "results.html",
        preview_table=preview_html,
        download_link=url_for("download_file", filename=filename),
        total=total,
        churned=churned,
        avg_prob=avg_prob,
        agg_recommendation=agg_recommendation
    )

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

@app.route("/dashboard")
def dashboard():
    global latest_predictions_path, model
    if latest_predictions_path is None or not os.path.exists(latest_predictions_path):
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
    return render_template(
        "dashboard.html",
        total=total,
        churn_count=churn_count,
        no_churn_count=no_churn_count,
        avg_prob=avg_prob,
        pie_labels=json.dumps(pie_labels),
        pie_values=json.dumps(pie_values),
        probs=json.dumps(probs),
        feature_importance=feature_importance
    )

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
    return render_template(
        "recommendations.html",
        avg_prob=avg_prob,
        agg_recommendation=agg_recommendation
    )

@app.route("/about")
def about():
    project_text = (
        "Insurance companies operate in a highly competitive environment. "
        "With data from millions of customers, understanding churn is challenging. "
        "This tool helps predict churn using 16 anonymized features (feature_0..feature_15)."
    )
    return render_template("about.html", project_text=project_text)

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
            df_churn = df.sort_values("Churn_Probability", ascending=False).iloc[0]
            churn_model_outputs = (
                f"Sample churn prediction:\n"
                f"Churn Probability: {df_churn['Churn_Probability']}\n"
                f"Recommendation: {df_churn['Recommendation']}\n"
            )
        except Exception:
            pass
    context_prompt = (
        "You are an AI assistant for insurance churn analysis. "
        "You have access to the latest churn prediction results. "
        "When the user asks about churn, customer behavior, or prevention strategies, "
        "use the following prediction insights to inform your answer:\n"
        f"{churn_model_outputs}\n"
        "Answer in a specific, business-friendly style with next steps and clear explanations."
    )
    full_prompt = context_prompt + "\nUser: " + user_message
    try:
        response = gemini_model.generate_content(full_prompt)
        reply = response.text
    except Exception as e:
        reply = f"Sorry, the AI service is currently unavailable. ({e})"
    return jsonify({"reply": reply})

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
