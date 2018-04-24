import cv2
import numpy as np
import matplotlib.pyplot as plot

def LabelColor(data):
    height, width, channel = data.shape
    for i in range(height):
        for j in range(width):
            if data[i,j,0] == 1:
                data[i,j] = [255, 0, 0]
            elif data[i,j,0] == 2:
                data[i,j] = [0,0,255]
            elif data[i,j,0] == 3:
                data[i,j] = [0,255,0]
            elif data[i,j,0] == 4:
                data[i,j] = [255,0,255]
            elif data[i,j,0] == 5:
                data[i,j] = [0,255,255]
            elif data[i,j,0] == 6:
                data[i,j] = [255,128,0]

img = cv2.imread('/home/gaobiao/Documents/FCN.tensorflow/logs/vis/test/inp_34991994.png', cv2.IMREAD_COLOR)
gt = cv2.imread('/home/gaobiao/Documents/FCN.tensorflow/logs/vis/test/gt_34991994.png', cv2.IMREAD_COLOR)
pre = cv2.imread('/home/gaobiao/Documents/FCN.tensorflow/logs/vis/test/pred_34991994.png', cv2.IMREAD_COLOR)

LabelColor(gt)
LabelColor(pre)

# _img = cv2.resize(img, (1080,144), interpolation=cv2.INTER_AREA)
# _gt = cv2.resize(gt, (1080,144), interpolation=cv2.INTER_AREA)
# _pre = cv2.resize(pre, (1080,144), interpolation=cv2.INTER_AREA)

plot.subplot(311)
plot.imshow(img)
plot.subplot(312)
plot.imshow(pre)
plot.subplot(313)
plot.imshow(gt)
plot.show()
