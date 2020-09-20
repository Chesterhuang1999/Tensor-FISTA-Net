#             Load data
#          Xiaochen  Han & Qifan Huang
#            Sept 20 2020
#          hqf17@mails.tsinghua.edu.cn
#

import tensorflow.compat.v1 as tf
import scipy.io as sio
import numpy as np
import DefineParam as DP
import h5py


# Get param
pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, trainFile, testFile, maskFile, saveDir, modelDir, name = DP.get_param()

tf.disable_v2_behavior()
# Load training data
def load_train_data(mat73=True):
    # load training data
    if mat73 == True:                                                # if .mat file is too big, use h5py to load
        trainData = h5py.File(trainFile)
        trainLabel = np.transpose(trainData['labels'], [3, 2, 1, 0]) # h5py reader will read the data in reverse dimension, so transpose it back
    else:
        trainData = sio.loadmat(trainFile)
        trainLabel = trainData['labels']                             # labels

    # load mask data
    maskData = sio.loadmat(maskFile)
    phi = maskData['phi']                                            # mask

    del trainData, maskData
    return trainLabel, phi


# Load testing data
def load_test_data(mat73=True):
    if mat73 == True:
        testData = h5py.File(testFile)
        testLabel = np.transpose(testData['labels'], [3, 2, 1, 0])
    else:
        testData = sio.loadmat(testFile)
        testLabel = testData['labels']
    
    maskData = sio.loadmat(maskFile)
    phi = maskData['phi']

    del testData, maskData
    return testLabel, phi


# Compute essential variables
def pre_calculate(phi):
    Xinput = tf.placeholder(tf.float32, [None, pixel, pixel, nFrame])      # X0
    Xoutput = tf.placeholder(tf.float32, [None, pixel, pixel, nFrame])     # labels
    Yinput = tf.placeholder(tf.float32, [None, pixel, pixel, 1])           # measurement
    Phi = tf.constant(phi)                                                 # phi
    PhiT = Phi                                                             # transpose phi (actually, transpose phi == phi in tensor form)

    return Xinput, Xoutput, Phi, PhiT, Yinput













