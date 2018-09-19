# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:21:18 2018

@author: Rahul kumar
"""


import cv2,os
import numpy as np
from PIL import Image

path="dataset"
recognizer = cv2.face.recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("C:\\Users\\Rahul kumar\\faceRecognitionProject\\Cascades\\haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:

        # Updates in Code
        # ignore if the file does not have jpg extension :
        if(os.path.split(imagePath)[-1].split(".")[-1]!='jpg'):
            continue

        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        
        #getting the Id from the image
        
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        
        # extract the face from the training image sample
        
        faces=detector.detectMultiScale(imageNp)
        
        #If a face is there then append that in the list as well as Id of it
        
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')
