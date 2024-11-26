import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from pathlib import Path
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)


class classify_video():
    def __init__(self):
        self.df=pd.read_csv(os.path.join(Path(__file__).parent,'data/processed_data.csv'))
        self.actions=['defense','kick','punch','sit','standing','walking']
        self.label=self.df['y']
        self.df=pd.get_dummies(self.df)
        self.x=self.df.iloc[:,:35]
        self.y=self.df.iloc[:,35:]
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.3)

    def build(self):
        input_layer=tf.keras.layers.Input((self.x.shape[1],))
        dense_layer=tf.keras.layers.Dense(200,activation='leaky_relu')(input_layer)
        dense_layer=tf.keras.layers.Dense(100,activation='leaky_relu')(dense_layer)
        dense_layer=tf.keras.layers.Dense(50,activation='leaky_relu')(dense_layer)
        dense_layer=tf.keras.layers.Dense(20,activation='leaky_relu')(dense_layer)
        dense_layer=tf.keras.layers.Dense(10,activation='leaky_relu')(dense_layer)
        dense_layer=tf.keras.layers.Dense(5,activation='leaky_relu')(dense_layer)
        dense_layer=tf.keras.layers.Dense(2,activation='leaky_relu')(dense_layer)
        output_layer=tf.keras.layers.Dense(len(self.actions),activation='softmax')(dense_layer)
        model=tf.keras.Model(input_layer,output_layer)
        
        return model

    def train(self,epochs=50,optimizer='adam',loss='categorical_crossentropy',batch_size=32):
        self.model=self.build()
        self.model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        best_lost=tf.keras.callbacks.ModelCheckpoint(os.path.join(Path(__file__).parent,'video.weights.h5'),save_weights_only=True,monitor='loss',mode='min',save_best_only=True)
        self.model.fit(self.x_train,self.y_train,epochs=epochs,batch_size=batch_size,callbacks=[best_lost])
        self.y_pred=self.model.predict(self.x)
    
    def evaluate(self):
        self.y_pred=list(map(lambda x:self.actions[x],np.argmax(self.y_pred,axis=1)))
        self.accuracy=accuracy_score(self.label,self.y_pred)
        self.precision=precision_score(self.label,self.y_pred,average='micro')
        self.recall=recall_score(self.label,self.y_pred,average='micro')
        self.f1=f1_score(self.label,self.y_pred,average='micro')
        
        return pd.Series([self.accuracy,self.precision,self.recall,self.f1],index=['accuracy','precision','recall','f1'])

    def heat_map(self):
        self.confusion_matrix=confusion_matrix(self.label,self.y_pred)
        self.confusion_matrix=np.round(self.confusion_matrix/self.confusion_matrix.sum(axis=1).reshape(-1,1),3)

        fg=plt.figure()
        ax=fg.add_subplot()
        ax.imshow(self.confusion_matrix,cmap='Blues')
        ax.grid(False)
        setting=[range(self.confusion_matrix.shape[0]),self.actions]
        ax.set_xticks(*setting)
        ax.set_yticks(*setting)
        for y in range(self.confusion_matrix.shape[0]):
            for x in range(self.confusion_matrix.shape[1]):
                color='white' if self.confusion_matrix[y,x]>=0.5 else 'black'
                ax.text(x,y,f'{self.confusion_matrix[y,x]}',ha='center',color=color)
        ax.set_title('Video actions confusion matrix')
        ax.set_ylabel('Y True')
        ax.set_xlabel('Y Predict')
        plt.tight_layout()
        plt.show()

if __name__=='__main__':
    video=classify_video()
    video.train()
    print(video.evaluate())
    video.heat_map()