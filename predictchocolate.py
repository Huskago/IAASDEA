import iaasdea

iaasdea = iaasdea.IAASDEA()

emotion = iaasdea.getPredictFromVideo("video.mkv")

print(emotion)