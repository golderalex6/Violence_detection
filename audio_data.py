from functional import *

class audio_data(audio_feature_extract):
    def __init__(self,n_mfcc=40):
        self.__n_mfcc=n_mfcc
        self.__audio_data_path=os.path.join(Path(__file__).parent,'data','audio_data')
        self.__labels=os.listdir(self.__audio_data_path)

    def process(self):
        x,y=[],[]
        label_encode={}
        for i in range(len(self.__labels)):
            files=[os.path.join(self.__audio_data_path,self.__labels[i],file) for file in os.listdir(os.path.join(self.__audio_data_path,self.__labels[i]))]
            label_encode[f"{self.__labels[i]}"]=i
            for file in files:
                audio, sample_rate = librosa.load(file)
                x.append(self._features_extractor(audio,sample_rate,self.__n_mfcc))
                y.append(i)

        x,y=np.array(x),np.array(y)
        df=pd.DataFrame(x)
        df['y']=y
        df.to_csv(os.path.join(Path(__file__).parent,'data','audio_processed_data.csv'),index=False)

        with open(os.path.join(Path(__file__).parent,'encode','audio_encode.json'),'w+') as f:
            json.dump(label_encode,f,indent=4)

if __name__=='__main__':
    data=audio_data()
    data.process()
