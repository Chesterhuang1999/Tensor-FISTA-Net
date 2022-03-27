#        Reconstruction Image
#           Xiaochen  Han
#            Apr 26 2019
#    guillermo_han97@sjtu.edu.cn
#

import scipy.io as sio
import numpy as np
from time import time
from PIL import Image
import math
import LoadData as LD
import DefineParam as DP
import os
from skimage import metrics 

# get param
pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, trainFile, testFile, maskFile, saveDir, modelDir, name = DP.get_param()

# load and form test image
def reconstruct_image(sess, Yinput, prediction, Xinput, Xoutput, testLabel, testPhi):
    # for initialization
    sumPhi = np.divide(1, np.maximum(np.sum(testPhi, axis=2), 1)).tolist()

    # check reconstructed images saving dir
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # pre-process
    xoutput = testLabel                                                                                    # labels
    yinput = np.sum(np.multiply(xoutput, testPhi), axis=3)                                                 # measurement
    xinput = np.tile(np.reshape(np.multiply(yinput, sumPhi), [-1, pixel, pixel, 1]), [1, 1, 1, nFrame])     # initialization
    yinput = np.reshape(yinput, (-1, pixel, pixel, 1))                                                     # measurement in correct shape
    feedDict = {Xinput: xinput, Xoutput: xoutput, Yinput: yinput}

    # do reconstruction
    start = time()
    rec = sess.run(prediction[-1], feed_dict=feedDict)
    end = time()

    # calculate psnr
    PSNR = psnr(xoutput, rec)
    recInfo = "Rec avg PSNR %.4f time= %.2fs\n" % (PSNR, (end - start))
    print(recInfo)
    # calculate ssim
    SSIM = metrics.structural_similarity(xoutput, rec, win_size = 3)
    recInfo1 = "Rec avg SSIM %.4f time= %.2fs\n" % (SSIM, (end - start))
    print(recInfo1)
    # output reconstruction image
    for i in range(3):
        meanpsnr = 0.00
        meanssim = 0.00
        for j in range(nFrame):
            PSNR = psnr(rec[i, :, :, j], xoutput[i, :, :, j])
            ssim = metrics.structural_similarity(rec[i,:,:,j],xoutput[i,:,:,j], win_size = 3)
            print("Frame %d %d, PSNR: %.2f" % (i, j, PSNR))
            meanpsnr += PSNR
            meanssim += ssim
            outImg = np.hstack((xoutput[i, :, :, j], rec[i, :, :, j]))
            imgRecName = "%s/frame%d channel%d PSNR%.2f.png" % (saveDir, i,j, PSNR)
            imgRec = Image.fromarray(np.clip(255*outImg, 0, 255).astype(np.uint8))
            imgRec.save(imgRecName)
        print("%d th image mean psnr %.2f ssim %.4f" %(i, meanpsnr/nFrame,meanssim/nFrame))
    sess.close()


# calculate psnr
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    return 20*math.log10(1.0/math.sqrt(mse))
