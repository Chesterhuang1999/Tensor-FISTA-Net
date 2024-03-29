#          Define all param
#          Xiaochen  Han & Qifan Huang
#          Sept 20 2020
#          hqf17@mails.tsinghua.edu.cn
#

# define all param
def get_param():    
    pixel = 256                # size of frames (pixel*pixel)
    batchSize = 10              # num of data in a batch
    nPhase = 8                # num of phases of Tensor FISTA-Net
    nTrainData = 500          # num of training data, decided by the training data file
    trainScale = 0.9           # scale of training part and validating part
    learningRate = 0.0002
    nEpoch = 600               # num of training epoch
    nFrame = 8               # num of frames compressed into one measurement (in paper is B)
    ncpkt = nEpoch             # reconstruction used model
    name = 'Vehicle'              # data set name
    trainFile = './trainData/train%s%d.mat' % (name, pixel)  # training file dir
    testFile = './testData/test%s%d.mat' % (name, pixel)     # testing file dir
    maskFile = './maskData/mask%d.mat' % pixel               # mask file dir
    saveDir = './recImg/recImg%d' % pixel                    # reconstruction save dir
    modelDir = './Model%d' % pixel                           # model save dir

    return pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, trainFile, testFile, maskFile, saveDir, modelDir, name
