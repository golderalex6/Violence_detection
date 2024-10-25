import cv2
import numpy as np
from pathlib import Path
import os
from ultralytics import YOLO
from train import classify_video


class pred_video():
    def __init__(self):
        self.model_yolo=YOLO(os.path.join(Path(__file__).parent,'yolov8n-pose.pt'))
        self.model=classify_video().build()
        self.model.load_weights(os.path.join(Path(__file__).parent,'video.weights.h5'))

    def normalize_point(self,box,points):
        x_min,y_min,x_max,y_max=box
        dist_x=x_max-x_min
        dist_y=y_max-y_min
        points[:,0]=(points[:,0]-x_min)/dist_x
        points[:,1]=(points[:,1]-y_min)/dist_y
        points=points.flatten()
        points=np.append(points,dist_y/dist_x)
        return points

    def pred_person(self,box,points):
        normalized = self.normalize_point(box,points)
        pred=self.model.predict(normalized.reshape(1,-1))[0]
        actions=['defense','kick','punch','sit','standing','walking']
        return actions[np.argmax(pred)]
    
    def predict(self,path):
        cap = cv2.VideoCapture(path)
        while True:
            ret,frame=cap.read()
            if ret:
                results=self.model_yolo(frame,conf=0.5)[0]
                for result in results:
                    boxes=result.boxes.numpy().xyxy.astype(int)
                    points=result.keypoints.numpy().xy
                    for i in range(len(boxes)):
                        label=self.pred_person(boxes[i],points[i])
                        cv2.putText(frame,label,(boxes[i][0],boxes[i][1]),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),thickness=5)
                        cv2.rectangle(frame,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(0,0,0),4)
                frame=cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
                cv2.imshow('real',frame)
            else:break
            if cv2.waitKey(1) == ord('k'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    path=os.path.join(Path(__file__).parent,'data/violence/defense_0.mp4')
    video=pred_video()
    video.predict(path)