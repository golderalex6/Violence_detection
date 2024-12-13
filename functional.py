import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path
import json
from abc import ABC,abstractmethod 
import typing


import librosa

import cv2
from ultralytics import YOLO

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split

plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)

class ModelBase:
    def __init__(
            self,
            encode_path:str,
            metadata_path:str,
            data_path:str,
            model_path:str
        ) -> None:
        '''
        Docstring here
        '''


        with open(encode_path,'r+') as f:
            self._label_encode = json.load(f)
            self._labels =sorted(list(self._label_encode.keys()),key=lambda x:self._label_encode[x])

        with open(metadata_path,'r+') as f:
            self._metadata = json.load(f)

        self._data_path = data_path
        self._model_path = model_path

    def _load_data(self):
        df = pd.read_csv(self._data_path)
        x = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        self._x_train,self._x_test,self._y_train,self._y_test  =  train_test_split(x,y,test_size=0.3)

    def _build(self) -> tf.keras.Model:
        layers = list(map(lambda x:tf.keras.layers.Dense(x,activation=self._metadata['activation']),self._metadata['layers']))
        model = tf.keras.Sequential([
                        tf.keras.layers.Input([self._x_train.shape[1]]),
                        *layers,
                        tf.keras.layers.Dense(len(self._labels),activation = 'softmax')
                    ])
        
        return model

    def train(self) -> None:
        self._load_data()

        self._model = self._build()
        self._model.compile(optimizer = self._metadata['optimizer'],loss=self._metadata['loss'],metrics=['accuracy'])
        best_lost = tf.keras.callbacks.ModelCheckpoint(self._model_path,monitor='loss',mode='min',save_best_only=True)
        self._model.fit(self._x_train,self._y_train,epochs = self._metadata['epochs'],batch_size=self._metadata['batch_size'],callbacks=[best_lost])

    def load(self,path:str = '') -> None:

        if path == '':
            self._model = load_model(self._model_path)
        else:
            self._model = load_model(path)

    def predict(self,x:typing.Iterable) -> int:
        
        return np.argmax(self._model.predict(x),axis = 1)


    def evaluate(self) -> None:

        self._load_data()
        y_pred = self.predict(self._x_test)

        accuracy = accuracy_score(self._y_test,y_pred)
        precision = precision_score(self._y_test,y_pred,average='micro')
        recall = recall_score(self._y_test,y_pred,average='micro')
        f1 = f1_score(self._y_test,y_pred,average='micro')

        print(f'Accuracy : {round(accuracy,2)}')
        print(f'Precision : {round(precision,2)}')
        print(f'Recall : {round(recall,2)}')
        print(f'F1 : {round(f1,2)}')
        
        matrix = confusion_matrix(self._y_test,y_pred)
        matrix = np.round(matrix/matrix.sum(axis=1).reshape(-1,1),2)

        fg = plt.figure()
        ax = fg.add_subplot()
        ax.imshow(matrix,cmap = 'Blues')
        ax.set_xticks(range(len(self._labels)),self._labels)
        ax.set_yticks(range(len(self._labels)),self._labels)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                color = 'white' if matrix[y,x]>=0.5 else 'black'
                ax.text(x,y,f'{matrix[y,x]}',ha = 'center',color=color)
        ax.set_title('Audio sound Confusion matrix')
        ax.set_ylabel('Y True')
        ax.set_xlabel('Y Predict')
        ax.grid(False)
        plt.show()

class AudioFeatureExtract:
    def __init__(self):
        self._N_MFCC = 40

    def _features_extractor(self,audio:np.ndarray,sample_rate:float) -> np.ndarray:
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self._N_MFCC)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        return mfccs_scaled_features

class VideoFeatureExtract:

    def _features_extractor(self,box:np.ndarray,points:np.ndarray) -> np.ndarray:
        x_min,y_min,x_max,y_max=box
        dist_x=x_max-x_min
        dist_y=y_max-y_min
        points[:,0]=(points[:,0]-x_min)/dist_x
        points[:,1]=(points[:,1]-y_min)/dist_y
        points=np.append(points.flatten(),dist_y/dist_x)
        return points
