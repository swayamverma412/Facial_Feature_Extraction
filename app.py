import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pandas as pd
import os
from collections import Counter
from flask import Flask, request, jsonify , render_template
import time

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh

# Load the saved models
model = tf.keras.models.load_model('model/fine_tuned_model.h5')
face_shape_model = tf.keras.models.load_model('model/FACE_SHAPE.h5')
gender_model = tf.keras.models.load_model('model/Gender_detection_model_new.h5')
skin_tone_model = tf.keras.models.load_model('model/skin_tone_model.keras')

class_names = ['Enfeksiyonel', 'Rosacea', 'Vitiligo', 'acne', 'clear skin', 'melasma']
face_shape_classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
gender_class = ['Female', 'Male']
skin_tone = ['Dark', 'Light', 'Mid-Dark', 'Mid-Light']

def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def calculate_length(landmarks, connections):
    total_length = 0
    for connection in connections:
        point1 = landmarks[connection[0]]
        point2 = landmarks[connection[1]]
        length = np.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)
        total_length += length
    return total_length

def extract_face_features(image):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        features = []

        for landmark in face_landmarks.landmark:
            features.append([landmark.x, landmark.y, landmark.z])

        return np.array(features).flatten()

def extract_features(image):
    forehead_connection = [(54, 104), (104, 69), (69, 108), (108, 151), (151, 337), (337, 299), (299, 333), (333, 284)]
    facelength_connection = [(10, 151), (151, 9), (9, 8), (8, 168), (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1), (1, 164), (164, 0), (0, 17), (17, 18), (18, 200), (200, 199), (199, 175), (175, 152)]
    jawline_connection = [(58, 214), (214, 43), (43, 16), (16, 273), (273, 434), (434, 288)]
    cheek_connection = [(234, 227), (227, 118), (118, 47), (47, 277), (277, 347), (347, 447), (447, 454)]

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            forehead_length = calculate_length(face_landmarks, forehead_connection)
            facelength_length = calculate_length(face_landmarks, facelength_connection)
            jawline_length = calculate_length(face_landmarks, jawline_connection)
            cheek_length = calculate_length(face_landmarks, cheek_connection)
            return [forehead_length, facelength_length, jawline_length, cheek_length], forehead_length, facelength_length, jawline_length, cheek_length
        else:
            return [0, 0, 0, 0], 0, 0, 0, 0

def predict_face_shape(image):
    features, forehead_length, facelength_length, jawline_length, cheek_length = extract_features(image)
    processed_image = preprocess_image(image, target_size=(220, 220))
    prediction = face_shape_model.predict([processed_image, np.array([features])])
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = face_shape_classes[predicted_class_index]
    return predicted_class_index, predicted_class_name, forehead_length, facelength_length, jawline_length, cheek_length

def predict_gender(image):
    processed_image = preprocess_image(image, target_size=(220, 220))
    prediction = gender_model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = gender_class[predicted_class_index]
    return predicted_class_index, predicted_class_name

def predict_skin_disease(image):
    processed_image = preprocess_image(image, target_size=(220, 220))
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_index, predicted_class_name

def predict_skin_tone(image):
    face_features = extract_face_features(image)
    if face_features is not None:
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = skin_tone_model.predict([np.array([face_features]), processed_image])
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = skin_tone[predicted_class_index]
        return predicted_class_index, predicted_class_name
    else:
        return None, None
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        predicted_class_index, predicted_class_name = predict_skin_disease(image)
        predicted_face_shape_class_index, predicted_face_shape_class_name, forehead_length, facelength_length, jawline_length, cheek_length = predict_face_shape(image)
        predicted_gender_class_index, predicted_gender_class_name = predict_gender(image)
        predicted_skin_tone_class_index, predicted_skin_tone_class_name = predict_skin_tone(image)
        
        result = {
            "skin_disease": predicted_class_name,
            "face_shape": predicted_face_shape_class_name,
            "gender": predicted_gender_class_name,
            "skin_tone": predicted_skin_tone_class_name,
            "forehead_length": forehead_length * 22.63,
            "face_length": facelength_length * 22.63,
            "jawline_length": jawline_length * 22.63,
            "cheek_length": cheek_length * 22.63
        }
        
        save_to_excel(result)
        
        return jsonify(result)
    else:
        return jsonify({"error": "No image uploaded"}), 400

def save_to_excel(data):
    filename = 'predictions.xlsx'
    df = pd.DataFrame([data])
    
    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
        
    df.to_excel(filename, index=False)

if __name__ == '__main__':
    app.run(debug=True)
