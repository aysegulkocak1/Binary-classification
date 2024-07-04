import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Classification:
    def __init__(self) :
        self.EPOCHS = 200
        self.TEST_SIZE = 0.33
        self.RANDOM_STATE = 42
        self.datas= load_breast_cancer()
        self.scaler= StandardScaler()


    def get_model(self,D):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Input(D,),
        tf.keras.layers.Dense(1,activation='sigmoid'),
        ])

        model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])

        return model
    def classificate(self):
        x_train,x_test,y_train,y_test = train_test_split(self.datas.data,self.datas.target,test_size=self.TEST_SIZE,random_state=self.RANDOM_STATE)
        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)
        model = self.get_model(x_train.shape[1])
        history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs= self.EPOCHS)

        print("Train score:", model.evaluate(x_train, y_train))
        print("Test score:", model.evaluate(x_test, y_test))
        self.plotGraphics(history)
        
        model.save("digitmodels.h5") 


    def plotGraphics(self,history):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()




obj = Classification()
obj.classificate()