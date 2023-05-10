from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model

# Définir les chemins vers les dossiers contenant les données d'entrainement et de validation
train_dir = "dataset/train"
val_dir = "dataset/test"

# Définir les paramètres d'entrainement du modèle
batch_size = 32 # Nombre d'images utilitées pour une itération du gradient descendant
epochs = 100 # Nombre d'itérations sur l'ensemble des données d'entrainement
input_shape = (48, 48, 1) # Dimensions des images d'entrée (largeur, hauteur, nombre de canaux)
num_classes = 7 # Nomre de classes de sortie (mis à jour avec les nouvelles classes ajoutées)

# Prétraiter les images en les redimensionnant et en les normalisant
train_datagen = ImageDataGenerator(rescale=1./255) # Applique une mise à l'échelle à chaque pixel pour normaliser les valeurs de chaque canal de couleur entre 0 et 1
train_generator = train_datagen.flow_from_directory(train_dir, target_size=input_shape[:2], color_mode='grayscale', batch_size=batch_size, class_mode='categorical', shuffle=True) # Génère des lots d'images de taille batch_size, redimensionnées à target_size et converties en niveau de gris, avec des étiquettes catégorielles pour chaque classe

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=input_shape[:2], color_mode='grayscale', batch_size=batch_size, class_mode='categorical', shuffle=True)

# Définir l'architecture du modèle DCNN pour la détection d'émotion
inputs = Input(shape=input_shape)
# block 1
x = Conv2D(32, kernel_size=3, activation="relu", padding="same")(inputs)
x = Conv2D(32, kernel_size=3, activation="relu", padding="same")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# block 2
x = Conv2D(64, kernel_size=3, activation="relu", padding="same")(x)
x = Conv2D(64, kernel_size=3, activation="relu", padding="same")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# block 3
x = Conv2D(128, kernel_size=3, activation="relu", padding="same")(x)
x = Conv2D(128, kernel_size=3, activation="relu", padding="same")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# block 4
x = Conv2D(256, kernel_size=3, activation="relu", padding="same")(x)
x = Conv2D(256, kernel_size=3, activation="relu", padding="same")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# flatten and dropout
x = Flatten()(x)
x = Dropout(0.5)(x)
# dense layers
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation="softmax")(x)

# Créer le modèle de réseau de neurones
model = Model(inputs, outputs)

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Définit l'optimiseur, la fonction de perte et les métriques à utiliser pendant l'entraînement

# Entrainement du modèle
history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)  # Entraîne le modèle avec les données d'entraînement générées par le générateur train_generator, et valide avec les données générées par val_generator

# Sauvegarder le modèle
model.save('deep_face_detection_model.h5') # Enregistre le modèle au format H5 pour une utilisation ultérieure