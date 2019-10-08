
# coding: utf-8

# In[ ]:


import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import face_recognition
from math import sqrt


# #### Libraries Used:
# The following libraries have been used in the entire code: 
# 1. Numpy
# 2. Matplotlib
# 3. OpenCV
# 4. Python Imaging Library (PIL)
# 5. face_recognition (This might be required to be installed through cmd/conda prompt. Prior to installing face_recognition, dlib has to be installed.) 
# 
# Steps: 
# 1. pip install dlib-19.8.1-cp36-cp36m-win_amd64.whl (File attached with the mail)
# 2. pip install face_recognition
# 
# 
# #### Pre-trained models used:
# Two pre-trained models have been used - one for face detection, and other for face recognition.
# ###### 1. Face-detection using Haar-Cascade classifiers:
# This works on the similar grounds as CNN and uses 6000+ features which are applied in different stages in small windows during the training. I've directly used the pre-trained classifier for face by the name of 'haarcascade_frontalface_default.xml' which will have to be downloaded beforehand. For more details, visit https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
# 
# ###### 2. Face-recognition using face_recognition library:
# In this we first derive feature vectors corresponding to a known image in the form of encodings and match them later with a new image. It employs dlib – a modern C++ toolkit that contains several machine learning algorithms. For more details, visit https://face-recognition.readthedocs.io/en/latest/face_recognition.html
# 
# 

# In[ ]:


haar_face_cascade = cv2.CascadeClassifier('C:/Users/HP/Downloads/haarcascade_frontalface_default.xml')


# In[ ]:


def convertToRGB(BGRimg): 
    return cv2.cvtColor(BGRimg, cv2.COLOR_BGR2RGB)


# In[ ]:


#Confidence score for face detection
def confidence_Score(faces,neigbors):
    numerator = faces-1
    denominator = neigbors-5
    confidence_score = (numerator/denominator)*300
    return confidence_score

