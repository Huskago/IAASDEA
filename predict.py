import cv2
from tensorflow import keras
import numpy as np
import time


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

# Réduire la résolution de l'image
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Initialiser le nombre de prédictions passées à prendre en compte
past_predictions = 10
prediction_list = []

# Boucle principale
while True:
    # Capturer une image
    ret, frame = cap.read()
    if not ret:
        print("Unable to capture video")
        break
        
    # Réduire la taille de l'image
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Classer chaque visage détecté
    for (x, y, w, h) in faces:
        # Extraire l'image du visage
        face_img = gray[y:y+w, x:x+w]
        
        # Redimensionner l'image pour l'entrée du modèle
        resized = cv2.resize(face_img, input_shape[:2], interpolation=cv2.INTER_AREA)
    
        # Prétraiter l'image pour la passer en entrée du modèle
        img_tensor = keras.preprocessing.image.img_to_array(resized)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        
        # Prédire la classe de l'image du visage
        prediction = model.predict(img_tensor)

        avg_prediction = np.mean(prediction_list)

        # Afficher la classe prédite en fonction de la moyenne
        if avg_prediction > 0.5:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Bad", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            # Afficher un rectangle vert autour du visage si la classe prédite est 'Good'
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Good", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        
#        # Afficher la classe prédite sur la frame
#        if prediction[0][0] > prediction[0][1]:
#            # Afficher un rectangle rouge autour du visage si la classe prédite est 'Bad'
#            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#            cv2.putText(frame, "Bad", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#        else:
#            # Afficher un rectangle vert autour du visage si la classe prédite est 'Good'
#            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#            cv2.putText(frame, "Good", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # Afficher la frame
    cv2.imshow('Face Classification', frame)
    
    # Quitter le programme si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
