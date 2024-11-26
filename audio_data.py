from functional import *

class audio_data():
    def __init__(self,n_mfcc=40):
        self.__n_mfcc=n_mfcc

        self.__audio_data_path=os.path.join(Path(__file__).parent,'data','audio_data')
        # non_violence_path=os.path.join(self.__audio_data_path,'non_violence')
        # violence_path=os.path.join(self.__audio_data_path,'violence')

        # self.__violence_files=[os.path.join(violence_path,file) for file in os.listdir(violence_path)]
        # self.__non_violence_files=[os.path.join(non_violence_path,file) for file in os.listdir(non_violence_path)]
        self.__labels=os.listdir(self.__audio_data_path)

    def _features_extractor(self,file):
        audio, sample_rate = librosa.load(file)
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.__n_mfcc)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        return mfccs_scaled_features
    
    def process(self):
        # self.x,self.y=[],[]
        # for i in self.__violence_files:
        #     self.x.append(self._features_extractor(i))
        #     self.y.append(1)
        # for i in self.__non_violence_files:
        #     self.x.append(self._features_extractor(i))
        #     self.y.append(0)
        # self.x,self.y=np.array(self.x),np.array(self.y)
        # self.data=pd.DataFrame(self.x,columns=np.arange(self.__n_mfcc)+1)
        # self.data['y']=self.y
        # self.data.to_csv(os.path.join(self.__audio_data_path,'processed_data.csv'),index=False)
        x,y=[],[]
        for label in self.__labels:
            files=[os.path.join(self.__audio_data_path,label,file) for file in os.listdir(os.path.join(self.__audio_data_path,label))]
            for file in files:
                x.append(self._features_extractor(file))
                y.append(label)
        x,y=np.array(x),np.array(y)
        df=pd.DataFrame(x)
        df['y']=y
        df.to_csv(os.path.join(Path(__file__).parent,'data','audio_processed_data.csv'),index=False)

data=audio_data()
data.process()
