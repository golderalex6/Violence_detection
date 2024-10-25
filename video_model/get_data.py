from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path

__root__=Path(__file__).parent

class get_data():
    def __init__(self):
        self.model=YOLO(os.path.join(__root__,'yolov8n-pose.pt'))

    def normalize_point(box,points):
        x_min,y_min,x_max,y_max=box
        dist_x=x_max-x_min
        dist_y=y_max-y_min
        points[:,0]=(points[:,0]-x_min)/dist_x
        points[:,1]=(points[:,1]-y_min)/dist_y
        points=points.flatten()
        points=np.append(points,dist_y/dist_x)
        return points
    
    def check_label(self,path):
        actions=['punch','kick','defense','sit','standing','walking']
        file=path.split('/')[-1]
        for action in actions:
            if file.startswith(action):
                return action
    def process(self,label='violence'):
        x,y=[],[]
        labels=['violence','non_violence']
        for label in labels:
            files=os.listdir(os.path.join(__root__,f'data/{label}'))
            files=list(map(lambda x:os.path.join(__root__,f'data/{label}/')+x,files))
            for file in files:
                cap=cv2.VideoCapture(file)
                l=0
                while l<=100:
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
                    l+=1
                    print(l)
                cap.release()
                cv2.destroyAllWindows()

        self.df=pd.DataFrame(x,columns=np.arange(len(x[0]))+1)
        self.df['y']=y
        self.df.to_csv(os.path.join(__root__,'data/processed_data.csv'),index=False)

data=get_data()
data.process()