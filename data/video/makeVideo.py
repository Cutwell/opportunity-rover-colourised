"""
Take all images from the {imageDir} directory and compile them into a video
"""

import cv2
import os

# Variables to change
frameRate = 10
imageDir = "../data/opportunity/stitch"
# - - - - - - - - - -

videoName = f"opportunity-rover-colourised-{frameRate}fps.avi"
imageNames = sorted([img for img in os.listdir(imageDir) if img.endswith(".png")])

frame = cv2.imread(os.path.join(imageDir, imageNames[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(videoName, 0, frameRate, (width, height))

for imageName in imageNames:
    print(f"Processing {imageName}")
    video.write(cv2.imread(os.path.join(imageDir, imageName)))

cv2.destroyAllWindows()
video.release()