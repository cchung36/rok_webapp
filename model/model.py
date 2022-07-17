import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input

class CNN(Model):
    def __init__(self,num_classes=10):
        super(CNN, self).__init__()
        self.conv1_64=Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu')
        self.conv2_64=Conv2D(64, (3, 3), activation='relu')
        self.maxpool1_22=MaxPooling2D(pool_size=(2,2))
        self.bn1=BatchNormalization()
        self.conv1_128=Conv2D(128, (3, 3), activation='relu')
        self.conv2_128=Conv2D(128, (3, 3), activation='relu')
        self.maxpool2_22=MaxPooling2D(pool_size=(2,2))
        self.bn2=BatchNormalization()
        self.conv1_256=Conv2D(256, (3, 3), activation='relu')
        self.maxpool3_22=MaxPooling2D(pool_size=(2,2))
        self.bn3=BatchNormalization()
        self.flatten=Flatten()
        self.dense = Sequential([
            Dense(512, activation='relu'),
            Dense(num_classes, activation='softmax')
        ]
        )

    def call(self,inputs, training=False, **kwargs):
        x=self.conv1_64(inputs)
        x=self.conv2_64(x)
        x=self.maxpool1_22(x)
        x=self.bn1(x, training=training)
        x=self.conv1_128(x)
        x=self.conv2_128(x)
        x=self.maxpool2_22(x)
        x=self.bn2(x, training=training)
        x=self.conv1_256(x)
        x=self.maxpool3_22(x)
        x=self.bn3(x, training=training)
        x=self.flatten(x)
        output=self.dense(x)

        return output

    def build_graph(self,raw_shape):
        x = Input(shape=raw_shape)
        return Model(inputs=[x], outputs=self.call(x))