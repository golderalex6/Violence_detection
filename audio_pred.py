import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import tensorflow as tf
from pathlib import Path
import os
from train import classify_audio

model=classify_audio().build()
model.load_weights(os.path.join(Path(__file__).parent,'audio.weights.h5'))

def features_extractor(audio,sample_rate):
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features,axis=1)
    return mfccs_scaled_features

audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1  
RATE = 22050
CHUNK = 1024
stream = audio.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)

def update_plot(data,value):
    if value>0.5:
        value=1
    else:
        value=0
    label={1:'Scream',0:'Non scream'}
    plt.clf()
    plt.plot(data)
    plt.title(label[value])
    plt.ylim(-1, 1)
    plt.pause(0.01)

try:
    plt.ion()
    while True:
        data = stream.read(CHUNK)
        audio_data=np.frombuffer(data, dtype=np.int16)
        audio_data=audio_data.astype(float)/32768.0
        extracted_features=features_extractor(audio_data,RATE)
        pred=model.predict(extracted_features.reshape(1,-1))[0]
        print(pred)
        update_plot(audio_data,pred)

except KeyboardInterrupt:
    print("Stopped recording.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
    plt.ioff()  
    plt.show()
