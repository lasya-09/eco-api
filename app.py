from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("eco_model_trained.h5")
class_names = ['cardboard', 'e-waste', 'glass', 'metal', 'organic', 'paper', 'plastic']

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_3": sorted(
            zip(class_names, predictions), key=lambda x: x[1], reverse=True
        )[:3]
    })

if __name__ == "__main__":
    app.run()
