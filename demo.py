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

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, input_shape=(256, 256, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dropout(rate=0.45))
model.add(Dense(67, activation='softmax'))

model.load_weights('cat_weight.h5')

def resize_image(image, target_size):
    # 獲取圖片的原始寬度和高度
    height, width, _ = image.shape
    
    # 計算目標寬度和高度
    target_width, target_height = target_size
    
    # 計算填充或裁剪的寬度和高度
    if width > height:
        diff = (width - height) // 2
        padding = [(diff, diff), (0, 0), (0, 0)]
    else:
        diff = (height - width) // 2
        padding = [(0, 0), (diff, diff), (0, 0)]
    
    # 對圖片進行填充或裁剪
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    
    # 將圖片調整為目標大小
    resized_image = cv2.resize(padded_image, target_size)
    
    return resized_image

# 讀取圖片
image = cv2.imread('cropped_image0.jpg')

# 設定目標大小
target_size = (256, 256)

# 將圖片調整為正方形並調整大小
resized_image = resize_image(image, target_size)

input_image = np.expand_dims(resized_image, axis=0)
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions, axis=1)
print("the cat is : ",class_names[predicted_class[0].astype(int)])