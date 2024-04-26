from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
model = load_model('model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (32, 32))
        image = image.reshape(1, 32, 32, 1)  # Reshape to match model input shape
    except Exception as e:
        return jsonify({'error': 'Error processing image'}), 400

    # Perform prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Define class names based on your model
    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    result = class_names[predicted_class]

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
