from iaasdea import IAASDEA

iaasdea = IAASDEA()
emotion = iaasdea.getPredictFromVideo("video.mkv")

print(emotion)