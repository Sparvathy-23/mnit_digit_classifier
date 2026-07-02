from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def simple_predict(image_path):
    """
    Lightweight deterministic classifier (no ML libraries).
    Simulates inference using image statistics.
    """

    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))

    arr = np.array(img)

    # Extract simple features
    mean_val = np.mean(arr)
    std_val = np.std(arr)

    # Deterministic mapping to digit (0–9)
    digit = int((mean_val + std_val) % 10)

    confidence = round(60 + (std_val % 40), 2)

    return digit, confidence


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="No file selected")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    prediction, confidence = simple_predict(filepath)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=filepath
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
