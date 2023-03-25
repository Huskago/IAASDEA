import cv2
from tensorflow import keras
import numpy as np


# Initialiser le détecteur de cascade Haar pour la détection de visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Charger le modèle préalablement entrainé
model = keras.models.load_model('face_classification_model.h5')

# Définir la taille d'entrée attendue pour le modèle
input_shape = (48, 48, 1)

# Définir les classes d'images
class_names = ['Bad', 'Good']

# Configurer la caméra
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to capture video")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y+w, x:x+w]
        resized = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_AREA)
        img_tensor = keras.preprocessing.image.img_to_array(resized)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        
        prediction = model.predict(img_tensor)
        if prediction[0][0] > prediction[0][1]:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Bad", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Good", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
