from flask import Flask, render_template, request, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Make sure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Waste categories
waste_types = {
    'cardboard': 'Recyclable',
    'glass': 'Recyclable',
    'metal': 'Recyclable',
    'paper': 'Recyclable',
    'plastic': 'Recyclable & Non-Biodegradable',
    'Trash': 'Biodegradable'
}

# Load the model
model = tf.keras.models.load_model('model/trained_model_for_waste_classification.keras')
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'Trash']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No image uploaded.")

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('index.html', error="No file selected.")

    # Save uploaded image
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)

    # Process the image
    image = Image.open(filepath).convert('RGB')
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)


    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = classes[predicted_class_index]
    predicted_category = waste_types.get(predicted_class_name, 'Unknown')
    confidence = predictions[0][predicted_class_index] * 100

    image_url = url_for('static', filename=f'uploads/{filename}')

    return render_template(
        'index.html',
        predicted_class=predicted_class_name,
        category=predicted_category,
        confidence=confidence,
        image_url=image_url
    )

if __name__ == '__main__':
    app.run(debug=True)
