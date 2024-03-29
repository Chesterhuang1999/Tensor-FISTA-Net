#      Train 2st Model of SCI
#           Xiaochen  Han
#            Apr 26 2019
#    guillermo_han97@sjtu.edu.cn
#

import LoadData as LD
import BuildModel as BM
import TrainModel as TM
import DefineParam as DP
import os 

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
# get param
pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, trainFile, testFile, maskFile, saveDir, modelDir, name = DP.get_param()

# load data
print('-------------------------------------\nLoading Data...\n-------------------------------------\n')
trainLabel, trainPhi = LD.load_train_data(mat73=True)  # if train data file is large and is mat7.3 version, set mat73=True
#trainLabel, trainPhi = LD.load_train_data(mat73=False)


# build model
print('-------------------------------------\nBuilding Model...\n-------------------------------------\n')
sess, saver, Xinput, Xoutput, costAll, optmAll, Yinput, prediction = BM.build_model(trainPhi)

# train data
print('-------------------------------------\nTraining Model...\n-------------------------------------\n')
TM.train_model(sess, saver, costAll, optmAll, Yinput, prediction, trainLabel, trainPhi, Xinput, Xoutput)

print('-------------------------------------\nTraining Accomplished.\n-------------------------------------\n')







