import cv2
import numpy as np
import tensorflow as tf
import time
import mediapipe as mp
import pandas as pd
import os
from collections import Counter

mp_face_mesh = mp.solutions.face_mesh

# Load the saved models
model = tf.keras.models.load_model('model/fine_tuned_model.h5')
face_shape_model = tf.keras.models.load_model('model/FACE_SHAPE.h5')
gender_model = tf.keras.models.load_model('model/Gender_detection_model_new.h5')
skin_tone_model = tf.keras.models.load_model('model/skin_tone_model.keras')

print("Model Loaded")

class_names = ['Enfeksiyonel', 'Rosacea', 'Vitiligo', 'acne', 'clear skin', 'melasma']
face_shape_classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
gender_class = ['Female', 'Male']
skin_tone = ['Dark', 'Light', 'Mid-Dark', 'Mid-Light']

# Define function to preprocess input image
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

# Define function to extract face features
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

# Define function to extract features from image
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

camera = cv2.VideoCapture(0)
start_time = time.time()
user_name = input("Enter your name: ")

# Lists to store predictions
skin_disease_predictions = []
face_shape_predictions = []
gender_predictions = []
skin_tone_predictions = []

while True:
    ret, frame = camera.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > 30:
        break

    predicted_class_index, predicted_class_name = predict_skin_disease(frame)
    cv2.putText(frame, "Skin Disease: " + str(predicted_class_name), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    predicted_face_shape_class_index, predicted_face_shape_class_name, forehead_length, facelength_length, jawline_length, cheek_length = predict_face_shape(frame)
    cv2.putText(frame, "Face Shape: " + str(predicted_face_shape_class_name), (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    predicted_gender_class_index, predicted_gender_class_name = predict_gender(frame)
    cv2.putText(frame, "Gender: " + str(predicted_gender_class_name), (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

    predicted_skin_tone_class_index, predicted_skin_tone_class_name = predict_skin_tone(frame)
    cv2.putText(frame, "Skin Tone: " + str(predicted_skin_tone_class_name), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

    cv2.imshow('Frame', frame)

    skin_disease_predictions.append(predicted_class_name)
    face_shape_predictions.append(predicted_face_shape_class_name)
    gender_predictions.append(predicted_gender_class_name)
    skin_tone_predictions.append(predicted_skin_tone_class_name)

    print("Predicted Skin Disease:", predicted_class_name)
    print("Predicted Face Shape:", predicted_face_shape_class_name)
    print("Predicted Gender:", predicted_gender_class_name)
    print("Predicted Skin Tone:", predicted_skin_tone_class_name)
    print("Forehead Length:", forehead_length)
    print("Face Length:", facelength_length)
    print("Jawline Length:", jawline_length)
    print("Cheek Length:", cheek_length)

camera.release()
cv2.destroyAllWindows()

# Get the most common prediction for each category
final_skin_disease = Counter(skin_disease_predictions).most_common(1)[0][0]
final_face_shape = Counter(face_shape_predictions).most_common(1)[0][0]
final_gender = Counter(gender_predictions).most_common(1)[0][0]
final_skin_tone = Counter(skin_tone_predictions).most_common(1)[0][0]

# Save the final prediction
final_prediction = [[user_name, final_skin_disease, final_face_shape, final_gender, final_skin_tone , forehead_length* 22.63, facelength_length * 22.63, jawline_length * 22.63, cheek_length * 22.63]]

# Read existing data if file exists
file_path = 'predictions.xlsx'
if os.path.exists(file_path):
    existing_df = pd.read_excel(file_path)
    new_df = pd.DataFrame(final_prediction, columns=['Name', 'Skin Disease', 'Face Shape', 'Gender','Skin Tone' ,'Forehead Length', 'Face Length', 'Jawline Length', 'Cheek Length'])
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
else:
    combined_df = pd.DataFrame(final_prediction, columns=['Name', 'Skin Disease', 'Face Shape', 'Gender','Skin Tone' , 'Forehead Length', 'Face Length', 'Jawline Length', 'Cheek Length'])

combined_df.to_excel(file_path, index=False)

print("Final Predicted Skin Disease:", final_skin_disease)
print("Final Predicted Face Shape:", final_face_shape)
print("Final Predicted Gender:", final_gender)
print("Final Predicted Skin tone:", final_skin_tone)
