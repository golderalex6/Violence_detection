import pandas as pd
import numpy as np

import os
from pathlib import Path
import json

import cv2
from ultralytics import YOLO

from functional import VideoFeatureExtract

class VideoData(VideoFeatureExtract):
    def __init__(self):
        self._model=YOLO(os.path.join(Path(__file__).parent,'yolov8n-pose.pt'))

        self._video_data_path=os.path.join(Path(__file__).parent,'data','video_data')
        self._labels=os.listdir(self._video_data_path)

    def _extract_data(self,file):
        x=[]
        cap=cv2.VideoCapture(file)
        for _ in range(100):
            ret,frame=cap.read()
            frame=cv2.flip(frame,1)
            if ret:
                results=self._model(frame,conf=0.5)[0].cpu()
                boxes=results.boxes.numpy().xyxy.astype(int)
                points=results.keypoints.numpy().xy
                for i in range(len(boxes)):
                    x.append(self._features_extractor(boxes[i],points[i]))
                    cv2.rectangle(frame,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(0,0,0),2)
                frame=cv2.resize(frame,(frame.shape[1]//4,frame.shape[0]//4))
                cv2.imshow('data',frame)
            else:break
            if cv2.waitKey(1) == ord('k'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return x

    def process(self):
        x,y=[],[]
        label_encode={}
        for i in range(len(self._labels)):
            files=[os.path.join(self._video_data_path,self._labels[i],file) for file in  os.listdir(os.path.join(self._video_data_path,self._labels[i]))]
            label_encode[f"{self._labels[i]}"]=i
            for file in files:
                data=self._extract_data(file)
                x.extend(data)
                y.extend([i]*len(data))

        x,y=np.array(x),np.array(y)
        df=pd.DataFrame(x)
        df['y']=y
        df.to_csv(os.path.join(Path(__file__).parent,'data','video_processed_data.csv'),index=False)

        with open(os.path.join(Path(__file__).parent,'encode','video.json'),'w+') as f:
            json.dump(label_encode,f,indent=4)

if __name__=='__main__':
    video=VideoData()
    video.process()
