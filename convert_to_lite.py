# Importer les bibliothèques nécessaires
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.models import save_model
import tensorflow as tf
import numpy as np

# Charger le modèle TensorFlow
model = load_model('face_classification_model.h5')

# Convertir le modèle en format TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Sauvegarder le modèle TensorFlow Lite
with open('face_classification_model.tflite', 'wb') as f:
    f.write(tflite_model)