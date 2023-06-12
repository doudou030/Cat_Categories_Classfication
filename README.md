# Cat_Categories_Classfication(CCC)

## requirement
先下載一些需要用的套件與dataset

## code
- yolov4.py
  - 用了pre-train的yolov4 model去框出貓
- train_classfication.py
  - 自製簡易的convolution model去做cat classfication
- demo.py
  - 讀取圖片並預測貓咪是什麼種類
- colab/Cat_Categories_Classfication.ipynb
  - 可以直接丟上colab上跑，雖然要跑很久就是了  

## index
- yolov4 
  - IoU (Intersection over Union) = 0.56
- classfication_model
  - loss: 2.5512 - accuracy: 0.4369
  - val_loss: 2.7777 - val_accuracy: 0.4244

## Demo
![](https://github.com/doudou030/Cat_Categories_Classfication/blob/main/img/demo.png?raw=true)
![](https://github.com/doudou030/Cat_Categories_Classfication/blob/main/img/single_demo.png?raw=true)

## dataset
[coco dataset](https://cocodataset.org/#home)

[貓咪classfication dataset](https://www.kaggle.com/datasets/denispotapov/cat-breeds-dataset-cleared)
