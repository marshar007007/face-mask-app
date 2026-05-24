from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load model 
model = tf.keras.models.load_model("face_mask_model.h5")

# Class names matching your training setup
# 0 = with_mask, 1 = without_mask  (alphabetical order from image_dataset_from_directory)
class_names = ['with_mask', 'without_mask']

IMG_SIZE = (128, 128)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_image(img_path):
    # Load and resize to be the same as my notebook
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    # Convert to array and add batch dimension - no manual /255 needed
    # (model has Rescaling(1./255) built in as the first layer)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 3)

    # Predict with training=False → augmentation + dropout OFF
    score = float(model(img_array, training=False)[0][0])

    # score > 0.5 → without_mask (class 1)
    # score ≤ 0.5 → with_mask    (class 0)
    if score > 0.5:
        label      = class_names[1]   # without_mask
        confidence = score
    else:
        label      = class_names[0]   # with_mask
        confidence = 1 - score

    return label, confidence


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            pred, conf = predict_image(filepath)

            return render_template(
                'index.html',
                filename=file.filename,
                prediction=pred,
                confidence=round(conf * 100, 2)
            )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)