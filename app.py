from flask import Flask, request, render_template
import tensorflow as tf
import cv2
import numpy as np
import uuid
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ LOAD MODEL ONCE AT STARTUP
model = tf.keras.models.load_model('crack_defect_model.keras')
model.make_predict_function()   # 🔥 IMPORTANT for Gunicorn

classes = ['Flexural Crack', 'HoneyCombing', 'Shear Crack']

@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    result = None
    confidence = None

    if request.method == 'POST':
        file = request.files.get('image')

        if file:
            filename = f"{uuid.uuid4()}.jpg"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img, verbose=0)
            idx = np.argmax(pred)

            result = classes[idx]
            confidence = float(pred[0][idx]) * 100

    return render_template(
        "index.html",
        filename=filename,
        result=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run()
