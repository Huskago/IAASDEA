from tensorflow import keras
import cv2
import numpy as np
import urllib.request

from enum import Enum

class Emotion(Enum):
    Colere = 0
    Degout = 1
    Peur = 2
    Joie = 3
    Neutre = 4
    Tristesse = 5
    Surprise = 6
    
    def getName(emotion):
        if emotion == 0:
            return "Colere"
        elif emotion == 1:
            return "Degout"
        elif emotion == 2:
            return "Peur"
        elif emotion == 3:
            return "Joie"
        elif emotion == 4:
            return "Neutre"
        elif emotion == 5:
            return "Tristesse"
        elif emotion == 6:
            return "Surprise"
        else:
            return "Unknown"

class IAASDEA:
    input_shape = (48, 48, 1)

    def __init__(self, model=keras.models.load_model("model.h5"), input_shape=(48, 48, 1)):
        self.model = model
        self.input_shape = input_shape

    def setModel(self, model):
        self.model = keras.models.load_model(model)
        
    def setModelFromUrl(self, url):
        file_name = url.split("/")[-1]
        urllib.request.urlretrieve(url, file_name)
        self.setModel(file_name)

    def setInputShape(self, input_shape):
        self.input_shape = input_shape
    
    def getPredictFromVideo(self, video_path):
        video = cv2.VideoCapture(video_path)

        past_predictions = 10
        prediction_list = []

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("Unable to get video")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            resized = cv2.resize(gray, self.input_shape[:2], interpolation=cv2.INTER_AREA)
        
            img_tensor = keras.preprocessing.image.img_to_array(resized)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.
            
            prediction = self.model.predict(img_tensor)
            
            prediction_list.append(prediction)
            
            if len(prediction) > past_predictions:
                prediction_list.pop(0)
                
            avg_prediction = np.mean(prediction_list, axis=0)
            
            max_index = np.argmax(avg_prediction)

        emotion = Emotion(max_index)
        return emotion
     
    def getFirstFrameFromVideo(self, video_path):
        video = cv2.VideoCapture(video_path)
        ret, frame = video.read()
        if not ret:
            print("Unable to get video")
            return None
        return frame