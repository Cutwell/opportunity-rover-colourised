"""
Take all images from the {imageDir} directory and compile them into a video
"""

from natsort import natsorted
import cv2
import os

# Variables to change
frameRate = 10
imageDir = "../result"
# - - - - - - - - - -

videoName = f"journey-{frameRate}fps.avi"
imageNames = natsorted([img for img in os.listdir(imageDir) if img.endswith(".jpg")])

frame = cv2.imread(os.path.join(imageDir, imageNames[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(videoName, 0, frameRate, (width, height))

for imageName in imageNames:
    print(f"Processing {imageName}")
    video.write(cv2.imread(os.path.join(imageDir, imageName)))

cv2.destroyAllWindows()
video.release()