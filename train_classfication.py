# api_token = {"username":"your username","key":"your API key"}

import json
import zipfile
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input, AveragePooling2D, Activation,Conv2D, MaxPooling2D, BatchNormalization,Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

if not os.path.exists("/root/.kaggle"):
    os.makedirs("/root/.kaggle")
 
with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

'''
tf.keras.utils.image_dataset_from_directory()

input : directory/images

output : A tf.data.Dataset object
'''
# 固定input_shape大小
input_shape = (256, 256)  
batch_size = 32

# 設定input image
train_data = tf.keras.utils.image_dataset_from_directory(
          '/content/dataset/images',
          validation_split=0.2,
          seed=123,
          subset="training",
          label_mode='categorical',# int categorical
          image_size=input_shape,
          batch_size=batch_size
          )  
val_data = tf.keras.utils.image_dataset_from_directory(
          '/content/dataset/images',
          validation_split=0.2,
          seed=123,
          subset="validation",
          label_mode='categorical',# int categorical
          image_size=input_shape,
          batch_size=batch_size
          )  
watching_data = tf.keras.utils.image_dataset_from_directory(
          '/content/dataset/images',
          label_mode='int',# int categorical
          image_size=input_shape,
          batch_size=batch_size
          ) 

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, input_shape=(256, 256, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dropout(rate=0.45))
model.add(Dense(67, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
callback = EarlyStopping(monitor = "val_accuracy",patience = 5)
history=model.fit(train_data, validation_data=val_data, epochs=100, batch_size=256, verbose=1, callbacks = [callback])

model.save_weights('cat_weight.h5')