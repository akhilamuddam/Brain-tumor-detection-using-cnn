

import os
import numpy as np
import tensorflow as tf

from flask import Flask, request, render_template_string
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
MODEL_PATH = "models/brain_tumor_model.keras"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMAGE_SIZE = (150, 150)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained CNN model
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print(f"WARNING: Model not found at {MODEL_PATH}")


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Brain MRI Image Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 650px;
            margin: 50px auto;
            padding: 20px;
            background: #f4f7fb;
            color: #202b3c;
        }
        .card {
            background: white;
            padding: 28px;
            border-radius: 12px;
            box-shadow: 0 4px 18px #0001;
        }
        h1 { color: #174a7e; }
        button {
            background: #174a7e;
            color: white;
            border: 0;
            padding: 12px 20px;
            border-radius: 6px;
            cursor: pointer;
        }
        input { margin: 18px 0; }
        .result {
            margin-top: 20px;
            padding: 15px;
            background: #eaf4ff;
            border-radius: 8px;
        }
        .error { color: #b42318; }
        .note { font-size: 13px; color: #555; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Brain MRI Image Classifier</h1>
        <p>Upload an MRI image for model classification.</p>

        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image"
                   accept=".png,.jpg,.jpeg" required>
            <br>
            <button type="submit">Classify Image</button>
        </form>

        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}

        {% if prediction %}
            <div class="result">
                <h2>Model Output</h2>
                <p><strong>Prediction:</strong> {{ prediction }}</p>
                {% if confidence is not none %}
                    <p><strong>Model score:</strong>
                       {{ confidence }}%</p>
                {% endif %}
            </div>
        {% endif %}

        <p class="note">
            Educational demonstration only. This tool is not
            a medical diagnostic system. Consult a qualified
            medical professional for interpretation of MRI scans.
        </p>
    </div>
</body>
</html>
"""


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    # Must match the preprocessing used during model training.
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(image_path):
    if model is None:
        raise RuntimeError(
            "Trained model is missing. Add the model file at "
            "models/brain_tumor_model.keras."
        )

    image_array = preprocess_image(image_path)
    output = np.asarray(model.predict(image_array, verbose=0))

    # Assumes binary classification with one sigmoid output.
    if output.size != 1:
        raise ValueError(
            "This app expects a binary CNN model with one output. "
            "Adapt prediction handling for your model's output format."
        )

    score = float(output.reshape(-1)[0])

    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("The model returned an invalid score.")

    # Assumes sigmoid output: >= 0.5 means Tumor.
    label = "Tumor" if score >= 0.5 else "No Tumor"

    # This is a model score, not a medically reliable probability.
    return label, round(score * 100, 2)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or uploaded_file.filename == "":
            error = "Please select an image."
        elif not allowed_file(uploaded_file.filename):
            error = "Upload a PNG, JPG, or JPEG image."
        else:
            filename = secure_filename(uploaded_file.filename)
            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"], filename
            )

            try:
                uploaded_file.save(image_path)
                # Verify that the file is a readable image.
                with Image.open(image_path) as image:
                    image.verify()

                prediction, confidence = predict_image(image_path)

            except (OSError, ValueError) as exc:
                error = f"Could not process image: {exc}"
            except RuntimeError as exc:
                error = str(exc)
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)

    return render_template_string(
        HTML,
        prediction=prediction,
        confidence=confidence,
        error=error,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return {"error": "Provide an image file in the 'image' field."}, 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename or not allowed_file(uploaded_file.filename):
        return {"error": "Upload a PNG, JPG, or JPEG image."}, 400

    filename = secure_filename(uploaded_file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        uploaded_file.save(image_path)

        with Image.open(image_path) as image:
            image.verify()

        prediction, score = predict_image(image_path)

        return {
            "prediction": prediction,
            "model_score_percent": score,
            "disclaimer": "Educational use only; not a medical diagnosis."
        }

    except (OSError, ValueError) as exc:
        return {"error": f"Could not process image: {exc}"}, 400
    except RuntimeError as exc:
        return {"error": str(exc)}, 503
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    app.run(debug=True)
