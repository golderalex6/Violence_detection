import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path
import json
from abc import ABC,abstractmethod

import cv2
from ultralytics import YOLO

import librosa
import pyaudio

import tensorflow as tf
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split

plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)

class model:

    def train(self,epochs=50,optimizer='adam',loss='binary_crossentropy',batch_size=32):
        self.model=self.build()
        self.model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        best_lost=tf.keras.callbacks.ModelCheckpoint(os.path.join(Path(__file__).parent,'audio.weights.h5'),save_weights_only=True,monitor='loss',mode='min',save_best_only=True)
        self.model.fit(self.x_train,self.y_train,epochs=epochs,batch_size=batch_size,callbacks=[best_lost])
        self.y_pred=self.model.predict(self.x).reshape(-1)

    def evaluate(self):
        pass

class audio_feature_extract:

    def _features_extractor(self,audio,sample_rate,n_mfcc):
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        return mfccs_scaled_features

class video_feature_extract:

    def _normalize_point(self,box,points):
        x_min,y_min,x_max,y_max=box
        dist_x=x_max-x_min
        dist_y=y_max-y_min
        points[:,0]=(points[:,0]-x_min)/dist_x
        points[:,1]=(points[:,1]-y_min)/dist_y
        points=np.append(points.flatten(),dist_y/dist_x)
        return points
