from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path

class video_data():
    def __init__(self):
        self.model=YOLO(os.path.join(Path(__file__).parent,'yolov8n-pose.pt'))

        self.__video_data_path=os.path.join(Path(__file__).parent,'data','video_data')
        non_violence_path=os.path.join(self.__video_data_path,'non_violence')
        violence_path=os.path.join(self.__video_data_path,'violence')

        self.__violence_files=[os.path.join(violence_path,file) for file in os.listdir(violence_path)]
        self.__non_violence_files=[os.path.join(non_violence_path,file) for file in os.listdir(non_violence_path)]

    def normalize_point(self,box,points):
        x_min,y_min,x_max,y_max=box
        dist_x=x_max-x_min
        dist_y=y_max-y_min
        points[:,0]=(points[:,0]-x_min)/dist_x
        points[:,1]=(points[:,1]-y_min)/dist_y
        points=np.append(points.flatten(),dist_y/dist_x)
        return points
    
    def check_label(self,path):
        actions=['punch','kick','defense','sit','standing','walking']
        file=path.split('/')[-1]
        for action in actions:
            if file.startswith(action):
                return action
    
    def _features_extractor(self,file):
        x,y=[],[]
        cap=cv2.VideoCapture(file)
        for _ in range(100):
            ret,frame=cap.read()
            frame=cv2.flip(frame,1)
            if ret:
                results=self.model(frame,conf=0.5)[0]
                boxes=results.boxes.numpy().xyxy.astype(int)
                points=results.keypoints.numpy().xy
                for i in range(len(boxes)):
                    x.append(self.normalize_point(boxes[i],points[i]))
                    y.append(self.check_label(file))
                    cv2.rectangle(frame,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(0,0,0),2)
                frame=cv2.resize(frame,(frame.shape[1]//4,frame.shape[0]//4))
                cv2.imshow('data',frame)
            else:break
            if cv2.waitKey(1) == ord('k'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return x,y

    def process(self,label='violence'):
        x,y=[],[]
        labels=['violence','non_violence']
        for label in labels:
            files=os.listdir(os.path.join(__root__,f'data/{label}'))
            files=list(map(lambda x:os.path.join(__root__,f'data/{label}/')+x,files))
            for file in files:
                x,y=


        df=pd.DataFrame(x,columns=np.arange(len(x[0]))+1)
        df['y']=y
        df.to_csv(os.path.join(__root__,'data/processed_data.csv'),index=False)

data=get_data()
data.process()
