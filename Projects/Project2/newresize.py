# resize pokeGAN.py
import os
import cv2

src = "./sprites" #pokeRGB_black
dst = "./newresizedData" # resized
if not os.path.exists(dst):
    os.mkdir(dst)

for each in os.listdir(src):
    img = cv2.imread(os.path.join(src,each))
    img = cv2.resize(img,(32,32))
    cv2.imwrite(os.path.join(dst,each), img)
    
