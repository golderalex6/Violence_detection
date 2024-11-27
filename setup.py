import os
from pathlib import Path
import json
import sys
import subprocess

#create folder 'data' to store data
if not os.path.exists(os.path.join(Path(__file__).parent,'data')):
    os.mkdir(os.path.join(Path(__file__).parent,'data'))

#create folder 'encode' 
if not os.path.exists(os.path.join(Path(__file__).parent,'encode')):
    os.mkdir(os.path.join(Path(__file__).parent,'encode'))

#create audio_encode.json/video_encode.json for store label encode
if not os.path.exists(os.path.join(Path(__file__).parent,'encode','audio_encode.json')):
    open(os.path.join(Path(__file__).parent,'encode','audio_encode.json'),'a+').close()

if not os.path.exists(os.path.join(Path(__file__).parent,'encode','video_encode.json')):
    open(os.path.join(Path(__file__).parent,'encode','video_encode.json'),'a+').close()

#create folder 'model' to store trainned model
if not os.path.exists(os.path.join(Path(__file__).parent,'model')):
    os.mkdir(os.path.join(Path(__file__).parent,'model'))

#create folder 'metadata' to store model hyper:q
if not os.path.exists(os.path.join(Path(__file__).parent,'metadata')):
    os.mkdir(os.path.join(Path(__file__).parent,'metadata'))

    #set default hyperparameter for model
    audio_hyperparameters={
            'layers':[100,50,20,10],
            'activation':'relu',
            'loss':'sparse_categorical_crossentropy',
            'optimizer':'adam',
            'epochs':10,
            'batch_size':32
        }
    with open(os.path.join(Path(__file__).parent,'metadata','audio_metadata.json'),'a+') as f:
        json.dump(audio_hyperparameters,f,indent=4)

    video_hyperparameters={
            'layers':[100,50,20,10],
            'activation':'relu',
            'loss':'sparse_categorical_crossentropy',
            'optimizer':'adam',
            'epochs':10,
            'batch_size':32
        }
    with open(os.path.join(Path(__file__).parent,'metadata','video_metadata.json'),'a+') as f:
        json.dump(video_hyperparameters,f,indent=4)

#install dataset if passed argument
if len(sys.argv)==2:
    if sys.argv[1]=='dataset':
        print('Start download the violence-nonviolence dataset !!')
        dataset_url='https://www.kaggle.com/api/v1/datasets/download/golderalex6/violence-nonviolence'
        subprocess.run([
                    f"curl -L -o {os.path.join(Path(__file__).parent,'archive.zip')} {dataset_url} &&\
                    unzip -d {os.path.join(Path(__file__).parent)} {os.path.join(Path(__file__).parent,'archive.zip')} && \
                    rm {os.path.join(Path(__file__).parent,'archive.zip')}"],shell=True)
    else:
        raise Exception("Invalid argument. Use 'dataset'!!")
