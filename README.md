# 🧠 MNIST Digit Classifier

A deep learning web application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN).

## Features

- Upload handwritten digit images
- CNN-based digit prediction
- Confidence score
- Image preview
- Modern responsive user interface
- Deployable on Render

## Tech Stack

- Python
- TensorFlow / Keras
- Flask
- HTML
- CSS

## Project Structure

```
mnist-digit-classifier/
│
├── app.py
├── mnist_cnn.keras
├── requirements.txt
├── README.md
├── .gitignore
│
├── screenshots/
│     └── Screenshot 2026-07-02 223430.png
│     └── Screenshot 2026-07-02 223509.png
│     └── Screenshot 2026-07-02 223546.png
│     └── render_screenshot.png
│
├── templates/
│     └── index.html
│
└── static/
      ├── style.css
      └── uploads/
```

## Dataset

The model is trained using the MNIST handwritten digit dataset.

## Model

- Convolutional Neural Network (CNN)
- ReLU activation
- MaxPooling
- Dropout
- Softmax Output Layer

## Author

Sree Parvathy
