
import os
import json
import re
import html
import sqlite3
import hashlib
import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sp

from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
ARTIFACT_DIR = os.path.join(BASE_DIR, "mlsecops_artifacts")
DB_PATH = os.path.join(BASE_DIR, "prediction_history.db")


# Artifact integrity verify karto
def verify_artifacts():
    manifest_path = os.path.join(ARTIFACT_DIR, "hash_manifest.json")

    with open(manifest_path, "r") as file:
        manifest = json.load(file)

    for filename, expected_hash in manifest.items():
        path = os.path.join(ARTIFACT_DIR, filename)

        if not os.path.isfile(path):
            raise RuntimeError(f"Missing artifact: {filename}")

        digest = hashlib.sha256()

        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                digest.update(chunk)

        if digest.hexdigest() != expected_hash:
            raise RuntimeError(f"Artifact integrity check failed: {filename}")


verify_artifacts()

final_model = joblib.load(os.path.join(ARTIFACT_DIR, "final_model.pkl"))
tfidf = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf.pkl"))
encoder = joblib.load(os.path.join(ARTIFACT_DIR, "encoder.pkl"))

text_columns = [
    "title", "company_profile", "description",
    "requirements", "benefits"
]

categorical_columns = [
    "location", "department", "salary_range",
    "employment_type", "required_experience",
    "required_education", "industry", "function"
]

binary_columns = [
    "telecommuting", "has_company_logo", "has_questions"
]


# SQLite database ani history table create karto
def init_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_details TEXT NOT NULL,
                prediction TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


init_database()


def save_prediction(job, prediction):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO prediction_history
            (job_details, prediction, created_at)
            VALUES (?, ?, ?)
            """,
            (
                json.dumps(job),
                prediction,
                datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
            )
        )


def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, job_details, prediction, created_at
            FROM prediction_history
            ORDER BY id DESC
            """
        ).fetchall()

    return rows


def get_history_item(item_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, job_details, prediction, created_at
            FROM prediction_history
            WHERE id = ?
            """,
            (item_id,)
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "job": json.loads(row["job_details"]),
        "prediction": row["prediction"],
        "created_at": row["created_at"]
    }


# Selected history record delete karto
def delete_history_item(item_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM prediction_history WHERE id = ?",
            (item_id,)
        )


# Complete prediction history delete karto
def clear_all_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM prediction_history")


def clean_text(text):
    text = str(text).lower()
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def validate_job(job):
    allowed_fields = set(text_columns + categorical_columns + binary_columns)

    if not isinstance(job, dict):
        raise ValueError("Invalid form submission.")

    for field, value in job.items():
        if field not in allowed_fields:
            raise ValueError("Unexpected input field.")

        if not isinstance(value, str):
            raise ValueError("Invalid input type.")

        if len(value) > 10000:
            raise ValueError("Input is too long.")

    for field in binary_columns:
        if job.get(field, "0") not in ("0", "1"):
            raise ValueError(f"Invalid value for {field}.")


def prepare_features(job):
    combined_text = " ".join(
        str(job.get(col, "")) for col in text_columns
    )
    cleaned_text = clean_text(combined_text)

    text_features = tfidf.transform([cleaned_text])

    categorical_data = pd.DataFrame(
        [{col: job.get(col, "") for col in categorical_columns}],
        columns=categorical_columns
    )
    categorical_features = encoder.transform(categorical_data)

    binary_data = np.array(
        [[int(job.get(col, "0")) for col in binary_columns]],
        dtype=np.float64
    )
    binary_features = sp.csr_matrix(binary_data)

    features = sp.hstack(
        [text_features, categorical_features, binary_features],
        format="csr"
    )

    if features.shape[1] != 9886:
        raise ValueError("Feature count mismatch.")

    return features


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    selected_item = None
    selected_id = request.args.get("history_id", type=int)

    if request.method == "POST":
        try:
            job = request.form.to_dict()
            validate_job(job)

            features = prepare_features(job)
            result = int(final_model.predict(features)[0])

            prediction = "FAKE JOB" if result == 1 else "REAL JOB"

            # Prediction database madhye save karto
            save_prediction(job, prediction)

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Prediction failed")
            error = "Unable to process this job. Please check your inputs."

    if selected_id is not None:
        selected_item = get_history_item(selected_id)

    history = get_history()

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        history=history,
        selected_item=selected_item
    )


# Ek selected history record delete karnyasathi POST route
@app.route("/history/delete/<int:item_id>", methods=["POST"])
def delete_history(item_id):
    delete_history_item(item_id)
    return redirect(url_for("home"))


# Complete history clear karnyasathi POST route
@app.route("/history/clear", methods=["POST"])
def clear_history():
    clear_all_history()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 5000)),
    debug=False
)
