import cv2
import os
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.image as mpimg
import pdb

SEG_PATH = '/home/gaobiao/Documents/FCN.tensorflow/Data_zoo/ladybug/seg/'
PATH = '/home/gaobiao/Documents/FCN.tensorflow/logs/vis/test_3channel_weight_lr5/'
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 144
NUM_OF_CLASSESS = 9

def ColorMap(data, img):
    if img[1] == 0:         # intensity == 0, invalid data
        data[:] = [img[0], img[0], img[0]]
        return data

    if data[0] == 1:        # people
        data = [0, 0,255]
    elif data[0] == 2:      # car
        data = [255,0,0]
    elif data[0] == 3:      # tree
        data = [0,255,0]
    elif data[0] == 4:      # sign
        data = [255,0,255]
    elif data[0] == 5:      # building
        data = [255,255,0]
    elif data[0] == 6:      # cyclist
        data = [0,128,255]
    elif data[0] == 7:      # stop bicycle
        data = [128,64,0]
    elif data[0] == 8:      # road
        data = [208,149,117]
    else:
        data[:] = [img[0], img[0], img[0]]
    return data

def LabelColor(img, gt, pre, seg):
    table = np.zeros((NUM_OF_CLASSESS, NUM_OF_CLASSESS))
    height, width, channel = gt.shape
    segImg = img.copy()
    for i in range(height):
        for j in range(width):
            x = int(i / 4.5)
            y = j * 2
            # caculate accuracy
            table[gt[i,j,0]][pre[i,j,0]] += 1
            # Label color of gt
            gt[i,j] = ColorMap(gt[i,j], img[i,j])
            # Label color of prediction
            pre[i,j] = ColorMap(pre[i,j], img[i,j])
            # Label color of segmentation result
            if seg[x,y] > 0:
                segImg[i,j] = ColorMap([seg[x,y] % (NUM_OF_CLASSESS-1) + 1], img[i,j])
            elif seg[x,y] == 0:
                segImg[i,j] = ColorMap([0], img[i,j])
            elif seg[x,y] < 0:
                segImg[i,j] = [0, 0, 0]
    return table, segImg

def DeleteLabel(gt, pre):
    height, width, channel = gt.shape
    for i in range(height):
        for j in range(width):
            if (gt[i,j,0] == 6):
                gt[i,j] = [1,1,1]
            if (pre[i,j,0] == 6):
                pre[i,j] = [1,1,1]

            if (gt[i,j,0] == 4):
                gt[i,j] = [3,3,3]
            if (pre[i,j,0] == 4):
                pre[i,j] = [3,3,3]

            if (gt[i,j,0] == 7):
                gt[i,j] = [3,3,3]
            if (pre[i,j,0] == 7):
                pre[i,j] = [3,3,3]
    return gt, pre

def OutputResult(table):
    fout = open('result_seg_lesslabel.txt', 'w')
    fout.write('%d\n' % NUM_OF_CLASSESS)
    for i in range(NUM_OF_CLASSESS):
        for j in range(NUM_OF_CLASSESS):
            fout.write('%d ' % table[i,j])
        fout.write('\n')
    fout.write('---------------------------\n')
    fout.write('label tp fp fn IoU\n')
    for i in range(1,NUM_OF_CLASSESS):
        tp = fp = fn = 0
        tp = table[i,i].astype(int)
        for j in range(1, NUM_OF_CLASSESS):
            if i != j:
                fp += table[j,i].astype(int)
                fn += table[i,j].astype(int)
        if (tp + fp + fn != 0):
            IoU = float(tp) / float(tp + fp + fn)
        else:
            IoU = 0
        fout.write('%d %d %d %d %.6f\n' % (int(i), int(tp), int(fp), int(fn), float(IoU)))
    fout.close()

def SegVote(seg, pre):
    dictVoteNum = dict()
    dictVoteId = dict()
    height, width, channel = pre.shape
    for i in range(height):
        for j in range(width):
            # if (i == 8 and j == 550):
            #     gb = 1
            x = int(i / 4.5)
            y = j * 2
            if (seg[x][y] > 0 and pre[i][j][0] != 0 and pre[i][j][0] != 8):
                if (not dictVoteId.has_key(seg[x][y])):
                    dictVoteId[seg[x][y]] = pre[i][j][0]
                    dictVoteNum[seg[x][y]] = 1
                elif (dictVoteId[seg[x][y]] == pre[i][j][0]):
                    dictVoteNum[seg[x][y]] += 1
                elif (dictVoteId[seg[x][y]] != pre[i][j][0]):
                    dictVoteNum[seg[x][y]] -= 1
                    if (dictVoteNum[seg[x][y]] <= -1):
                        dictVoteId[seg[x][y]] = pre[i][j][0]
                        dictVoteNum[seg[x][y]] = 1
    # update pre
    for i in range(height):
        for j in range(width):
            # if (i == 8 and j == 550):
            #     gb = 1
            x = int(i / 4.5)
            y = j * 2
            if (seg[x][y] > 0 and dictVoteId.has_key(seg[x][y])):
                # pre[i,j] = [1,1,1]
                pre[i,j] = [dictVoteId[seg[x][y]], dictVoteId[seg[x][y]], dictVoteId[seg[x][y]]]
            # elif seg[x][y] == -999:     # road
                # pre[i,j] = [8, 8, 8]
    return pre


listName = os.listdir(PATH)
imgList = []
gtList = []
preList = []
cntTable = np.zeros((NUM_OF_CLASSESS, NUM_OF_CLASSESS))

for name in listName:
    if name.split('/')[-1][0] == 'i':
        imgList.append(name)
    elif name.split('/')[-1][0] == 'g':
        gtList.append(name)
    elif name.split('/')[-1][0] == 'p':
        preList.append(name)

imgList.sort()
gtList.sort()
preList.sort()

videoWriter = cv2.VideoWriter('test_seg_lesslabel.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 5, (IMAGE_WIDTH, IMAGE_HEIGHT * 3), True)
#videoWriter = cv2.VideoWriter('test_seg.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (IMAGE_WIDTH, IMAGE_HEIGHT * 3), True)

for i in range(len(imgList)):
    img = cv2.imread(PATH + imgList[i], cv2.IMREAD_COLOR)
    gt = cv2.imread(PATH + gtList[i], cv2.IMREAD_COLOR)
    pre = cv2.imread(PATH + preList[i], cv2.IMREAD_COLOR)

    timeStamp =  imgList[i].split('_')[1].split('.')[0]
    f_seg = open(SEG_PATH + timeStamp + '_seg.txt', 'r')
    seg = f_seg.readline().strip().split(' ')
    seg = np.array(map(int, seg))
    height, width, channel = img.shape
    seg = seg.reshape(32, 2160)
    gt, pre = DeleteLabel(gt, pre)
    pre = SegVote(seg, pre)

    table, segImg = LabelColor(img, gt, pre, seg)
    cntTable += table
    mergeImg = np.concatenate((pre,gt,segImg), axis=0)
    cv2.putText(mergeImg, str(timeStamp), (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)
    videoWriter.write(mergeImg)
    print('Frame: %d' % i)

OutputResult(cntTable)

videoWriter.release()
