import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS 
from tf2_yolov4.model import YOLOv4
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2 
# print(cv2.__version__)
%matplotlib inline

WIDTH, HEIGHT = (640, 480) #選擇辨識畫面解析度

model = YOLOv4(
    input_shape=(HEIGHT, WIDTH, 3), #輸入影像規格
    anchors=YOLOV4_ANCHORS,      #使用YOLO設定的錨
    num_classes=80,          #辨識物件80種    
    yolo_max_boxes=50,         #最多找到50個
    yolo_iou_threshold=0.5,      #iou門檻0.5
    yolo_score_threshold=0.5,     #信任門檻0.5
)
 
model.load_weights('yolov4.h5') #請注意路徑是否正確
#YOLOv4所能辨識的物件列表 
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop',  'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

path = "1cat.jpg"#讀檔
img = cv2.imread(path)

#將cv2影像轉換成TF格式
img = cv2.resize(img,(WIDTH, HEIGHT))
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32) , axis=0) / 255

#進行物件偵測
boxes, scores, classes, null = model.predict(image)
#boxes物件在畫面的位置
boxes = boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT]
#scores物件的信賴程度
scores = scores[0]
#物件的名稱
classes = classes[0].astype(int)


#依序讀取偵測到的物件，並畫出框線
for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):
    if score > 0.5: #設定信任度>0.5才會顯示
        #畫出物件範圍
        x1 = int(xmin)
        x2 = int(xmax)
        y1 = int(ymin)
        y2 = int(ymax)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 178, 50), 3)
        #在物件範圍左上寫出物件名稱+信任度
        text = CLASSES[class_idx] + ': {0:.2f}'.format(score)            
        cv2.putText(img, text,  (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# 將框出來的貓裁下來
ori_img = cv2.imread(path)
re_img = img = cv2.resize(ori_img,(WIDTH, HEIGHT))
i = 0
for (xmin, ymin, xmax, ymax) in boxes:#最多只能跑50個
  x1 = int(xmin)
  x2 = int(xmax)
  y1 = int(ymin)
  y2 = int(ymax)
  crop_img = re_img[y1:y2, x1:x2]
  # 保存圖像

  if (x1 == 0) & (x2 == 0) & (y1 == 0) & (y2 == 0) :
    continue
  else:
    # 顯示圖片
    cv2.imwrite(f'cropped_image{i}.jpg', crop_img)
    crop_img = mpimg.imread(f'cropped_image{i}.jpg')
    plt.axis('off')
    plt.imshow(crop_img)
    plt.show()
    i = i + 1