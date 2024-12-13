import pandas as pd
import numpy as np

import os
from pathlib import Path
import json

import librosa

from functional import AudioFeatureExtract
class AudioData(AudioFeatureExtract):
    def __init__(self) -> None:
        super().__init__()

        self._audio_data_path=os.path.join(Path(__file__).parent,'data','audio_data')
        self._labels=os.listdir(self._audio_data_path)

    def process(self) -> None:

        x,y=[],[]
        label_encode={}
        for i in range(len(self._labels)):
            files=[os.path.join(self._audio_data_path,self._labels[i],file) for file in os.listdir(os.path.join(self._audio_data_path,self._labels[i]))]
            label_encode[f"{self._labels[i]}"]=i
            for file in files:
                audio, sample_rate = librosa.load(file)
                x.append(self._features_extractor(audio,sample_rate))
                y.append(i)

        x,y=np.array(x),np.array(y)
        df=pd.DataFrame(x)
        df['y']=y
        df.to_csv(os.path.join(Path(__file__).parent,'data','audio_processed_data.csv'),index=False)

        with open(os.path.join(Path(__file__).parent,'encode','audio.json'),'w+') as f:
            json.dump(label_encode,f,indent=4)

if __name__=='__main__':
    data=AudioData()
    data.process()
