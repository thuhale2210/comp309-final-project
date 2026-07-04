# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 03:31:17 2025

@author: hannah
"""

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

FEATURE_COLUMNS = [
    "PRIMARY_OFFENCE",
    "OCC_YEAR",
    "OCC_MONTH",
    "OCC_DOW",
    "OCC_DAY",
    "OCC_DOY",
    "OCC_HOUR",
    "REPORT_YEAR",
    "REPORT_MONTH",
    "REPORT_DOW",
    "REPORT_DAY",
    "REPORT_DOY",
    "REPORT_HOUR",
    "DIVISION",
    "LOCATION_TYPE",
    "PREMISES_TYPE",
    "BIKE_TYPE",
    "BIKE_SPEED",
    "BIKE_COST"
]

with open("bike_return_model.pkl", "rb") as f:
    bundle = pickle.load(f)

preprocessor = bundle["preprocessor"]
selector = bundle["selector"]
clf = bundle["classifier"]

app = Flask(__name__)


def make_predictions_from_records(records):
    """
    records: list[dict]
    Returns: list of dicts with prediction_label, prediction_class, probability_not_returned
    """
    data_df = pd.DataFrame(records)

    # Ensure all expected columns exist
    for col in FEATURE_COLUMNS:
        if col not in data_df.columns:
            data_df[col] = None

    # Order columns exactly as in training
    data_df = data_df[FEATURE_COLUMNS]

    # Preprocess using the same pipeline as training
    X_proc = preprocessor.transform(data_df)
    X_sel = selector.transform(X_proc)

    # Convert to dense if needed
    try:
        X_sel = X_sel.toarray()
    except AttributeError:
        pass

    # Predict probabilities: class 1 = NOT RETURNED / STOLEN
    probs = clf.predict_proba(X_sel)[:, 1]
    preds = (probs >= 0.3).astype(int)

    results = []
    for p, pr in zip(preds, probs):
        label = "NOT RETURNED (stolen)" if p == 1 else "RETURNED"
        results.append({
            "probability_not_returned": float(pr),
            "prediction_label": label,
            "prediction_class": int(p)
        })

    return results


@app.route("/", methods=["GET"])
def home():
    return "Bike Return Prediction API is running"


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API endpoint.

    Expects JSON with the 19 feature fields.
    Can accept:
      - a single JSON object (dict)
      - a list of JSON objects

    Returns:
      {
        "results": [
          {
            "prediction_label": "RETURNED" or "NOT RETURNED (stolen)",
            "prediction_class": 0 or 1,
            "probability_not_returned": float
          },
          ...
        ]
      }
    """
    try:
        input_json = request.get_json()

        # Allow both a single dict or a list of dicts
        if isinstance(input_json, dict):
            records = [input_json]
        else:
            records = input_json

        results = make_predictions_from_records(records)
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/ui", methods=["GET"])
def bike_form():
    """
    Render the HTML form to collect bike theft incident details.
    """
    return render_template("bike_form.html")


@app.route("/ui/result", methods=["POST"])
def bike_result():
    """
    Handle form submission from /ui and show a human-friendly result page.
    """
    form = request.form

    # Build record dict from form fields
    record = {
        "PRIMARY_OFFENCE": form.get("PRIMARY_OFFENCE") or None,
        "OCC_YEAR": form.get("OCC_YEAR") or None,
        "OCC_MONTH": form.get("OCC_MONTH") or None,
        "OCC_DOW": form.get("OCC_DOW") or None,
        "OCC_DAY": form.get("OCC_DAY") or None,
        "OCC_DOY": form.get("OCC_DOY") or None,
        "OCC_HOUR": form.get("OCC_HOUR") or None,
        "REPORT_YEAR": form.get("REPORT_YEAR") or None,
        "REPORT_MONTH": form.get("REPORT_MONTH") or None,
        "REPORT_DOW": form.get("REPORT_DOW") or None,
        "REPORT_DAY": form.get("REPORT_DAY") or None,
        "REPORT_DOY": form.get("REPORT_DOY") or None,
        "REPORT_HOUR": form.get("REPORT_HOUR") or None,
        "DIVISION": form.get("DIVISION") or None,
        "LOCATION_TYPE": form.get("LOCATION_TYPE") or None,
        "PREMISES_TYPE": form.get("PREMISES_TYPE") or None,
        "BIKE_TYPE": form.get("BIKE_TYPE") or None,
        "BIKE_SPEED": form.get("BIKE_SPEED") or None,
        "BIKE_COST": form.get("BIKE_COST") or None,
    }

    # Convert numeric fields where possible
    numeric_fields = [
        "OCC_YEAR", "OCC_DAY", "OCC_DOY", "OCC_HOUR",
        "REPORT_YEAR", "REPORT_DAY", "REPORT_DOY", "REPORT_HOUR",
        "BIKE_COST"
    ]
    for field in numeric_fields:
        value = record[field]
        if value not in (None, ""):
            try:
                record[field] = int(value)
            except ValueError:
                pass

    results = make_predictions_from_records([record])
    result = results[0]

    return render_template(
        "bike_result.html",
        input_data=record,
        prediction_label=result["prediction_label"],
        prediction_class=result["prediction_class"],
        probability_not_returned=result["probability_not_returned"]
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
