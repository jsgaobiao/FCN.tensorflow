import numpy as np
import cv2
import os
import matplotlib.pyplot as plot
import matplotlib.image as mpimg
import pdb
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

PATH = '/home/gaobiao/Documents/FCN.tensorflow/logs/vis/test_3channel_weight_lr5/'
NUM_OF_CLASSESS = 9
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 144

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

def LabelColor(img, gt, pre):
    table = np.zeros((NUM_OF_CLASSESS, NUM_OF_CLASSESS))
    height, width, channel = gt.shape
    for i in range(height):
        for j in range(width):
            # caculate accuracy
            if (img[i,j,0] < 660):
                # merge label
                if (gt[i,j,0] == 6):
                    gt[i,j,0] = 1
                if (pre[i,j,0] == 6):
                    pre[i,j,0] = 1
                if (gt[i,j,0] == 7):
                    gt[i,j,0] = 3
                if (pre[i,j,0] == 7):
                    pre[i,j,0] = 3
                table[gt[i,j,0]][pre[i,j,0]] += 1
                # Label color of gt
                gt[i,j] = ColorMap(gt[i,j], img[i,j])
                # Label color of prediction
                pre[i,j] = ColorMap(pre[i,j], img[i,j])
    return table

"""
Function which returns the labelled image after applying CRF

#Original_image = Image which has to labelled
#Annotated image = Which has been labelled by some technique( FCN in this case)
#Output_image = The final output image after applying CRF
#Use_2d = boolean variable
#if use_2d = True specialised 2D fucntions will be applied
#else Generic functions will be applied

"""
def crf(original_image, annotated_image, output_image, use_2d = True):

    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:,:,0] + (annotated_image[:,:,1]<<8) + (annotated_image[:,:,2]<<16)
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    # colors = np.array([0,1,2,3,4,5,6,7,8])
    # labels = annotated_label.flatten()
    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat))
    # print("No of labels in the Image are ")
    # print(n_labels)

    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
        # get unary potentials (neg log probability)
        zero_flag = True
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=zero_flag)
        d.setUnaryEnergy(U)
        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 130, 13), rgbim=original_image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    #Run Inference for 5 steps
    Q = d.inference(5)
    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)
    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    if (zero_flag):
        MAP = MAP + 1
    MAP = colorize[MAP,:]
    # imsave(output_image,MAP.reshape(original_image.shape))
    return MAP.reshape(original_image.shape)

def OutputResult(table, crfTable):
    fout = open('crf/result_test_3channel_weight_lr5_crf.txt', 'w')
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

    fout.write('-----------------CRF-----------------\n')
    for i in range(NUM_OF_CLASSESS):
        for j in range(NUM_OF_CLASSESS):
            fout.write('%d ' % crfTable[i,j])
        fout.write('\n')
    fout.write('---------------------------\n')
    fout.write('label tp fp fn IoU\n')
    for i in range(1,NUM_OF_CLASSESS):
        tp = fp = fn = 0
        tp = crfTable[i,i].astype(int)
        for j in range(1, NUM_OF_CLASSESS):
            if i != j:
                fp += crfTable[j,i].astype(int)
                fn += crfTable[i,j].astype(int)
        if (tp + fp + fn != 0):
            IoU = float(tp) / float(tp + fp + fn)
        else:
            IoU = 0
        fout.write('%d %d %d %d %.6f\n' % (int(i), int(tp), int(fp), int(fn), float(IoU)))
    fout.close()

# main
listName = os.listdir(PATH)
imgList = []
gtList = []
preList = []
cntTable = np.zeros((NUM_OF_CLASSESS, NUM_OF_CLASSESS))
crfTable = np.zeros((NUM_OF_CLASSESS, NUM_OF_CLASSESS))

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

# local PC
# videoWriter = cv2.VideoWriter('crf/test_crf.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 5, (IMAGE_WIDTH, IMAGE_HEIGHT * 2), True)
# Server
videoWriter = cv2.VideoWriter('crf/test_crf.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (IMAGE_WIDTH, IMAGE_HEIGHT * 3), True)

for i in range(len(imgList)):
    img = cv2.imread(PATH + imgList[i], cv2.IMREAD_COLOR)
    gt = cv2.imread(PATH + gtList[i], cv2.IMREAD_COLOR)
    pre = cv2.imread(PATH + preList[i], cv2.IMREAD_COLOR)
    gt3 = gt.copy()
    pre3 = pre.copy()

    table = LabelColor(img, gt3, pre3)
    cntTable += table

    output = crf(img, pre, "crf_%d.png" % i)
    table = LabelColor(img, gt, output)
    crfTable += table

    mergeImg = np.concatenate((pre3, output, gt), axis=0)
    # cv2.imshow("img", mergeImg)
    # cv2.waitKey(0)
    videoWriter.write(mergeImg)
    print('Frame: %d' % i)

OutputResult(cntTable, crfTable)

videoWriter.release()
