from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained CNN model
model = load_model("mnist_cnn.keras")


def preprocess_image(image_path):
    """
    Preprocess uploaded image for MNIST prediction
    """

    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))

    img = np.array(img)

    # Invert colors if background is white
    if np.mean(img) > 127:
        img = 255 - img

    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Please upload an image."
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="No image selected."
        )

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    processed = preprocess_image(filepath)

    prediction = model.predict(processed)

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    return render_template(
        "index.html",
        prediction=predicted_digit,
        confidence=round(confidence, 2),
        image_path=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)