import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path

import pyaudio
from functional import AudioFeatureExtract,ModelBase

class AudioModel(AudioFeatureExtract,ModelBase):

    def __init__(self) -> None:

        AudioFeatureExtract.__init__(self)

        ModelBase.__init__(
                self,
                os.path.join(Path(__file__).parent,'encode','audio.json'),
                os.path.join(Path(__file__).parent,'metadata','audio.json'),
                os.path.join(Path(__file__).parent,'data','audio_processed_data.csv'),
                os.path.join(Path(__file__).parent,'model','audio.keras')
            )
    
    def audio_predict(self) -> None:

        audio = pyaudio.PyAudio()
        RATE = 22050
        CHUNK = 1024
        stream = audio.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,frames_per_buffer=CHUNK)

        fg = plt.figure()
        ax = fg.add_subplot()
        while True:
            ax.cla()
            audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
            audio_data = audio_data.astype(float)/32768.0
            extracted_features = self._features_extractor(audio_data,RATE)
            label = self._labels[np.argmax(self._model.predict(extracted_features.reshape(1,-1)))]

            ax.set_ylim(-2, 2)
            ax.plot(audio_data)
            ax.set_title(label)
            plt.pause(0.01)
            if len(plt.get_fignums()) == 0:
                raise Exception('Turn off training')

if __name__ =='__main__':
    audio = AudioModel()
    # audio.train()
    audio.load()
    audio.evaluate()
    # audio.audio_predict()
