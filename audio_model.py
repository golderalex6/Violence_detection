from functional import *

class audio_model(audio_feature_extract):
    def __init__(self):
        self.df=pd.read_csv(os.path.join(Path(__file__).parent,'data/audio_processed_data.csv'))
        self.x=self.df.iloc[:,:-1]
        self.y=self.df.iloc[:,-1]
        self.__x_train,self.__x_test,self.__y_train,self.__y_test=train_test_split(self.x,self.y,test_size=0.3)

        with open(os.path.join(Path(__file__).parent,'encode','audio_encode.json'),'r+') as f:
            self.__label_encode=json.load(f)
            self.__labels=sorted(list(self.__label_encode.keys()),key=lambda x:self.__label_encode[x])

        with open(os.path.join(Path(__file__).parent,'metadata','audio_metadata.json')) as f:
            self.__metadata=json.load(f)

    def _load_data(self):
        pass

    def _build(self):
        layers=list(map(lambda x:tf.keras.layers.Dense(x,activation=self.__metadata['activation']),self.__metadata['layers']))
        model=tf.keras.Sequential([
                tf.keras.layers.Input((self.x.shape[1],)),
                *layers,
                tf.keras.layers.Dense(len(self.__labels),activation='softmax')
            ])
        
        return model

    def train(self):
        self.model=self._build()
        self.model.compile(optimizer=self.__metadata['optimizer'],loss=self.__metadata['loss'],metrics=['accuracy'])
        best_lost=tf.keras.callbacks.ModelCheckpoint(os.path.join(Path(__file__).parent,'model','audio_model.weights.h5'),save_weights_only=True,monitor='loss',mode='min',save_best_only=True)
        self.model.fit(self.__x_train,self.__y_train,epochs=self.__metadata['epochs'],batch_size=self.__metadata['batch_size'],callbacks=[best_lost])
    
    def evaluate(self):
        model=self._build()
        model.load_weights(os.path.join(Path(__file__).parent,'model','audio_model.weights.h5'))
        y_pred=np.argmax(model.predict(self.__x_test),axis=1)

        accuracy=accuracy_score(self.__y_test,y_pred)
        precision=precision_score(self.__y_test,y_pred,average='micro')
        recall=recall_score(self.__y_test,y_pred,average='micro')
        f1=f1_score(self.__y_test,y_pred,average='micro')

        print(f'Accuracy : {round(accuracy,2)}')
        print(f'Precision : {round(precision,2)}')
        print(f'Recall : {round(recall,2)}')
        print(f'F1 : {round(f1,2)}')
        
        matrix=confusion_matrix(self.__y_test,y_pred)
        matrix=np.round(matrix/matrix.sum(axis=1).reshape(-1,1),2)

        fg=plt.figure()
        ax=fg.add_subplot()
        ax.imshow(matrix,cmap='Blues')
        ax.set_xticks(range(len(self.__labels)),self.__labels)
        ax.set_yticks(range(len(self.__labels)),self.__labels)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                color='white' if matrix[y,x]>=0.5 else 'black'
                ax.text(x,y,f'{matrix[y,x]}',ha='center',color=color)
        ax.set_title('Audio sound Confusion matrix')
        ax.set_ylabel('Y True')
        ax.set_xlabel('Y Predict')
        ax.grid(False)
        plt.show()

    def audio_predict(self):

        model=self._build()
        model.load_weights(os.path.join(Path(__file__).parent,'model','audio_model.weights.h5'))

        audio = pyaudio.PyAudio()
        RATE = 22050
        CHUNK = 1024
        stream = audio.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,frames_per_buffer=CHUNK)

        fg=plt.figure()
        ax=fg.add_subplot()
        while True:
            ax.cla()
            audio_data=np.frombuffer(stream.read(CHUNK), dtype=np.int16)
            audio_data=audio_data.astype(float)/32768.0
            extracted_features=self._features_extractor(audio_data,RATE,40)
            label=self.__labels[np.argmax(model.predict(extracted_features.reshape(1,-1),verbose=False))]

            ax.set_ylim(-2, 2)
            ax.plot(audio_data)
            ax.set_title(label)
            plt.pause(0.01)
            if len(plt.get_fignums())==0:
                    raise Exception('Turn off training')

if __name__=='__main__':
    audio=audio_model()
    # audio.train()
    # audio.evaluate()
    audio.audio_predict()
