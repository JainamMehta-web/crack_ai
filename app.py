import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, render_template

# ----------------- CONFIG -----------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO/WARNING
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = None
try:
    import tensorflow as tf
    model = tf.keras.models.load_model('crack_defect_model.keras')
except Exception as e:
    print("Error loading model:", e)

# Class names (must match training order)
classes = ['Flexural Crack', 'HoneyCombing', 'Shear Crack']

# ----------------- ROUTES -----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    result = None
    confidence = None

    if request.method == 'POST':
        file = request.files.get('image')

        if file and file.filename:
            # Save uploaded image
            filename = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            try:
                # Preprocess image
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                # Predict
                pred = model.predict(img)
                class_index = np.argmax(pred)
                result = classes[class_index]
                confidence = float(pred[0][class_index]) * 100

            except Exception as e:
                result = f"Prediction Error: {str(e)}"
                confidence = None

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        filename=filename
    )

# ----------------- RUN -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
