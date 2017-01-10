# -*- coding: UTF-8 -*-

import cv2
import math
import numpy as np
import os
from PIL import Image

fixed_x, fixed_w = 112, 112

if __name__ == '__main__':
    cascade_path = "/home/mluser/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    
    #img_src = cv2.imread("./data/sample.jpg", 1)
    #img_src = cv2.imread("./data/sample2.png", 1)
        
    img_pil = Image.open('./data/sample.jpg')
    img_src = np.asarray(img_pil)

    #img_result = img_src
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    
    faces = cascade.detectMultiScale(img_gray, scaleFactor=1.1, \
                    minNeighbors=1, minSize=(10,10))
    
    if len(faces) > 0:
        color = (255, 0, 0)
        for face in faces:
            pos1 = tuple(face[0:2])                # 左上の座標
            pos2 = tuple(face[0:2] + face[2:4])   # 横と縦の長さ
            print("Found face: pos1, pos2= ", pos1, pos2)
            #cv2.rectangle(img_result, pos, pos2, color, thickness=2)
            
            margin = 1.2
            
            # size of face
            w = pos2[0] - pos1[0]
            h = pos2[1] - pos1[1]
            if w > h:
                shape = (int(fixed_w * margin), int((fixed_h * h / w) * margin))
            else:
                shape = (int(fixed_w * w / h * margin), int(fixed_w * margin))
                            
            pos1_x = pos1[0] - int((margin - 1) * w/2)
            pos1_y = pos1[1] - int((margin - 1) * h/2)
            pos2_x = pos2[0] + int((margin - 1) * w/2)
            pos2_y = pos2[1] + int((margin - 1) * h/2)
            
            img_pil2 = img_pil.crop((pos1_x, pos1_y, pos2_x, pos2_y))
            img_pil2 = img_pil2.resize(shape)
            img_result = np.asarray(img_pil2)
               
    cv2.imshow("Show FACES Image", img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
