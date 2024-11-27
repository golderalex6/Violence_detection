from functional import *

class video_data(video_feature_extract):
    def __init__(self):
        self.model=YOLO(os.path.join(Path(__file__).parent,'yolov8n-pose.pt'))

        self.__video_data_path=os.path.join(Path(__file__).parent,'data','video_data')
        self.__labels=os.listdir(self.__video_data_path)

    def _features_extractor(self,file):
        x=[]
        cap=cv2.VideoCapture(file)
        for _ in range(100):
            ret,frame=cap.read()
            frame=cv2.flip(frame,1)
            if ret:
                results=self.model(frame,conf=0.5)[0].cpu()
                boxes=results.boxes.numpy().xyxy.astype(int)
                points=results.keypoints.numpy().xy
                for i in range(len(boxes)):
                    x.append(self._normalize_point(boxes[i],points[i]))
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
        for i in range(len(self.__labels)):
            files=[os.path.join(self.__video_data_path,self.__labels[i],file) for file in  os.listdir(os.path.join(self.__video_data_path,self.__labels[i]))]
            label_encode[f"{self.__labels[i]}"]=i
            for file in files:
                data=self._features_extractor(file)
                x.extend(data)
                y.extend([i]*len(data))

        x,y=np.array(x),np.array(y)
        df=pd.DataFrame(x)
        df['y']=y
        df.to_csv(os.path.join(Path(__file__).parent,'data','video_processed_data.csv'),index=False)

        with open(os.path.join(Path(__file__).parent,'encode','video_encode.json'),'w+') as f:
            json.dump(label_encode,f,indent=4)

if __name__=='__main__':
    video=video_data()
    video.process()
