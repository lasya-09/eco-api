from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("eco_model_trained.h5")
class_names = ['cardboard', 'e-waste', 'glass', 'metal', 'organic', 'paper', 'plastic']

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['file']
        if not file:
            return jsonify({"error": "No file part"}), 400
        
        # Convert file to BytesIO object
        img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's port or default to 5000
    app.run(host='0.0.0.0', port=port)
