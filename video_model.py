import numpy as np

import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from functional import VideoFeatureExtract,ModelBase

class VideoModel(VideoFeatureExtract,ModelBase):
    def __init__(self):

        ModelBase.__init__(
                self,
                os.path.join(Path(__file__).parent,'encode','video.json'),
                os.path.join(Path(__file__).parent,'metadata','video.json'),
                os.path.join(Path(__file__).parent,'data','video_processed_data.csv'),
                os.path.join(Path(__file__).parent,'model','video.keras')
            )

    def video_predict(self,path:str = '') -> None:
        model_yolo=YOLO(os.path.join(Path(__file__).parent,'yolov8n-pose.pt'))

        cap = cv2.VideoCapture(path)
        while True:
            ret,frame=cap.read()
            if ret:
                results=model_yolo(frame,conf=0.5)[0].cpu()
                for result in results:
                    boxes=result.boxes.numpy().xyxy.astype(int)
                    points=result.keypoints.numpy().xy
                    for i in range(len(boxes)):
                        extracted_features = self._features_extractor(boxes[i],points[i])
                        label = self._labels[np.argmax(self._model.predict(extracted_features.reshape(1,-1)))]
                        cv2.putText(frame,label,(boxes[i][0],boxes[i][1]),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),thickness=5)
                        cv2.rectangle(frame,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(255,255,255),4)
                    frame = cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
                cv2.imshow('real',frame)
            else:break
            if cv2.waitKey(1) == ord('k'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    video=VideoModel()
    video.load()
    # video.train()
    # video.evaluate()
    video.video_predict('data/video_data/kick/kick_1.mp4')
