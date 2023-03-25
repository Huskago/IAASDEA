from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# Définir les chemins vers les dossiers contenant les données d'entrainement et de validation
train_dir = "dataset/train"
val_dir = "dataset/validation"

# Définir les paramètres d'entrainement du modèle
batch_size = 32
epochs = 100
input_shape = (48, 48, 1)
num_classes = 2

# Prétraiter les images en les redimensionnant et en les normalisant
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=input_shape[:2], color_mode='grayscale', batch_size=batch_size, class_mode='categorical', shuffle=True)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=input_shape[:2], color_mode='grayscale', batch_size=batch_size, class_mode='categorical', shuffle=True)

# Créer le modèle de réseau de neurones
model = keras.Sequential([
  keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(num_classes, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrainement du modèle
history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Sauvegarder le modèle
model.save('face_classification_model.h5')