import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"./model/")))
import cv2
import numpy as np
from model import CNN
import tensorflow as tf
from tensorflow.train import Checkpoint
from mnist import MNIST

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

dir1='/src/test/Screenshot_2022.07.07_03.42.17.816.png'
dir2='/src/test/Screenshot_2022.07.07_14.51.45.828.png'

img1 = cv2.cvtColor(cv2.imread(dir1),cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread(dir2),cv2.COLOR_BGR2GRAY)

#print("shape:{}".format(img1.shape))
#print("shape:{}".format(img2.shape))

model_dir='/src/checkpoint/model-1085.tf'
model = CNN()
model.load_weights(model_dir)
# # # test
# dir='/src/dataset'
# mndata=MNIST(dir)

# test_images, test_labels = mndata.load_testing()
# testX = tf.convert_to_tensor(test_images,dtype=tf.float32)
# testY = tf.convert_to_tensor(test_labels,dtype=tf.uint32)
# testX = testX/255.0

# testX = tf.reshape(testX,[testX.shape[0],28,28,1])

# test=testX[0]
# test=tf.expand_dims(test,axis=0)
# print('shape1:{}'.format(test.shape))

# prediction = model.predict(test)
# result = np.argmax(prediction)

# print("prediction:{}".format(result))
# print("label:{}".format(testY[0]))

# # # ---------------------------------------
width = 220
height = 50

x=1450
y=165

crop_img1=img1[y:y+height,x:x+width]
crop_img2=img2[y:y+height,x:x+width]

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen1 = cv2.filter2D(crop_img1, -1, kernel)
sharpen2 = cv2.filter2D(crop_img2, -1, kernel)

_, thresh1 = cv2.threshold(sharpen1, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresh2 = cv2.threshold(sharpen2, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def smooth(img,counter):
    ret = cv2.pyrUp(img)

    for i in range(counter):
        ret = cv2.medianBlur(ret,3)

    return cv2.pyrDown(ret)

smooth1 = smooth(thresh1,10)
smooth2 = smooth(thresh2,10)

_, bin1 = cv2.threshold(smooth1, 90, 255, cv2.THRESH_BINARY)
_, bin2 = cv2.threshold(smooth2, 90, 255, cv2.THRESH_BINARY)

filename = 'img1.jpg'
cv2.imwrite(filename, bin1)
filename = 'img2.jpg'
cv2.imwrite(filename, bin2)

def process_image(img,target):
    xdelta=10
    ydelta=14
    ret , labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    samples=[]
    for label in range(1,ret):
        mask = np.array(labels, dtype=np.uint8)
        temp = np.array(target, dtype=np.float32)
        if len(mask[labels == label]) > 50:
            mask[labels == label] = 255
            xc=int(centroids[label][0])
            yc=int(centroids[label][1])
            filename = 'img_'+str(label)+'.jpg'
            temp[labels != label] = 0.0
            ROI = temp[yc-ydelta:yc+ydelta,xc-xdelta:xc+xdelta]
            sample = cv2.copyMakeBorder(ROI, 0, 0, 4, 4, cv2.BORDER_CONSTANT, None, value = 0)
            samples.append(sample)
            cv2.imwrite(filename, sample)

    return samples

samples = process_image(bin1,smooth1)

for image in samples:
    inputs = tf.convert_to_tensor(image,dtype=tf.float32)
    inputs = tf.reshape(inputs,[1,28,28,1])
    inputs = inputs/255.0
    prediction = model.predict(inputs)
    result = tf.argmax(prediction, axis=-1)
    print("prediction:{}".format(result))