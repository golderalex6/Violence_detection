import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path

import tensorflow as tf
from ultralytics import YOLO
import librosa
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split

plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)

AUDIO_DATA=os.path.join(Path(__file__).parent,'data','audio_data')
VIDEO_DATA=os.path.join(Path(__file__).parent,'data','video_data')
