import cv2
from tensorflow import keras
import numpy as np


# Initialiser le détecteur de cascade Haar pour la détection de visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Charger le modèle préalablement entrainé
model = keras.models.load_model('model-best.h5')

# Définir la taille d'entrée attendue pour le modèle
input_shape = (48, 48, 1)

# Définir les classes d'images ('angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised')
class_names = ['Colere', 'Degout', 'Peur', 'Joie', 'Neutre', 'Tristesse', 'Surprise']

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
        
        # Ajouter la prédiction dans la liste
        prediction_list.append(prediction)
        
        if len(prediction) > past_predictions:
            prediction_list.pop(0)
            
        # Calculer la moyenne des prédictions
        avg_prediction = np.mean(prediction_list, axis=0)
        
        # Trouver la classe prédite avec la probabilité la plus élevée
        max_index = np.argmax(avg_prediction)
        predicted_class = class_names[max_index]
        
        # Afficher un rectangle coloré autour du visage en fonction de la classe prédite
        color = (0, 0, 0)
        if predicted_class == 'Colere':
            color = (0, 0, 255)
        elif predicted_class == 'Degout':
            color = (0, 128, 0)
        elif predicted_class == 'Peur':
            color = (255, 255, 0)
        elif predicted_class == 'Joie':
            color = (0, 255, 0)
        elif predicted_class == 'Neutre':
            color = (128, 128, 128)
        elif predicted_class == 'Tristesse':
            color = (255, 0, 0)
        elif predicted_class == 'Surprise':
            color = (255, 0, 255)
            
        # Calculer la taille du texte
        text_size = cv2.getTextSize(predicted_class, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

        # Calculer les coordonnées du coin supérieur gauche de la boîte de texte centrée
        text_x = x + (w - text_size[0]) // 2
        text_y = y - text_size[1] - 5
        
        # Dessiner la boîte de délimitation et le texte
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), color, 2)
        cv2.putText(frame, predicted_class, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Afficher la frame
    cv2.imshow('Face Classification', frame)
    
    # Quitter le programme si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
