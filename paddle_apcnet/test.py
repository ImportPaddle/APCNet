from tqdm import tqdm
import paddle
import datetime
import logging
import os
from dataloader import CityScapesDataset
from architectures import getApcNet
from paddle.io import Dataset,DataLoader
from  config import CONFIG
import paddle.nn.functional as F
import warnings
import glob 
import numpy as np
from architectures import getApcNet
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from cv2 import imread, imwrite
from PIL import Image
def getMiou(mat):
        ious=[]
        for i in range(19):
            iou_i=mat[i][i]/(np.sum(mat[i],axis=0)+np.sum(mat[:,i],axis=0)-mat[i][i])
            ious.append(iou_i)
        
        res=(np.mean(ious),ious)
        return res
def CRFs(original_image_path, predicted_image_path, CRF_image_path):

    img = imread(original_image_path)

    # 将predicted_image的RGB颜色转换为uint32颜色 0xbbggrr
    anno_rgb = imread(predicted_image_path).astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # 将uint32颜色转换为1,2,...
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # 如果你的predicted_image里的黑色（0值）不是待分类类别，表示不确定区域，即将分为其他类别
    # 那么就取消注释以下代码
    # HAS_UNK = 0 in colors
    # if HAS_UNK:
    # colors = colors[1:]

    # 创建从predicted_image到32位整数颜色的映射。
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 计算predicted_image中的类数。
    n_labels = len(set(labels.flat))
    # n_labels = len(set(labels.flat)) - int(HAS_UNK) ##如果有不确定区域，用这一行代码替换上一行

    ###########################
    ###     设置CRF模型     ###
    ###########################
    use_2d = False
    # use_2d = True
    ###########################################################
    ##不是很清楚什么情况用2D
    ##作者说“对于图像，使用此库的最简单方法是使用DenseCRF2D类”
    ##作者还说“DenseCRF类可用于通用（非二维）密集CRF”
    ##但是根据我的测试结果一般情况用DenseCRF比较对
    #########################################################33
    if use_2d:
        # 使用densecrf2d类
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 增加了与颜色无关的术语，功能只是位置而已
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 增加了颜色相关术语，即特征是(x,y,r,g,b)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        # 使用densecrf类
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 这将创建与颜色无关的功能，然后将它们添加到CRF中
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 这将创建与颜色相关的功能，然后将它们添加到CRF中
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ###         做推理和计算         ###
    ####################################

    # 进行5次推理
    Q = d.inference(5)

    # 找出每个像素最可能的类
    MAP = np.argmax(Q, axis=0)

    # 将predicted_image转换回相应的颜色并保存图像
    MAP = colorize[MAP, :]
    imwrite(CRF_image_path, MAP.reshape(img.shape))
    print("CRF图像保存在", CRF_image_path, "!")

def getConfusionMatrix(prediction,target,ignore_labeel=255):
    confusionMatrix=np.zeros((19,19),dtype=int)
    prediction=prediction.reshape(-1)
    target=target.reshape(-1)
    for (p1,p2) in zip(target,prediction):
        if p1!=255:
            confusionMatrix[p1,p2]+=1
    return confusionMatrix
def test():
    valDataset=CityScapesDataset(root='../dataset/cityscapes',mode='val',SEED=1)
    valLoader=DataLoader(valDataset,batch_size=1,drop_last=False,num_workers=4, shuffle=False,use_buffer_reader=True)
    models,msg_resnet=getApcNet()
    state=paddle.load('./architectures/pretrained/apcnet_r101-d8_512x1024_80k_cityscapes_20201214_115705-b1ff208a.paparams')
    models['backbone'].set_state_dict(state['models']['backbone'])
    models['APCHead'].set_state_dict(state['models']['APCHead'])
    models['FCNHead'].set_state_dict(state['models']['FCNHead'])
    
    models['backbone'].eval()
    models['APCHead'].eval()
    models['FCNHead'].eval()
    confusionMatrix=np.zeros((19,19))
    for i,batch in enumerate(tqdm(valLoader())):
            x,label=batch
            x=paddle.to_tensor(x,dtype='float32')
            # print(label.shape)
            label=paddle.to_tensor(label,dtype='int64')
            # print('label',label.shape)
            feature2=models['backbone'](x)[0]
            feature3=models['backbone'](x)[1] #0,1,2,3
            # print('feature.shape',feature.shape)
            pre1=models['APCHead'](feature3)
            # pre2=models['FCNHead'](feature2)
            pre1=F.interpolate(x=pre1, size=[512,1024])
            # pre2=F.interpolate(x=pre2, size=[512,1024])
            
            prediction=paddle.argmax(pre1,axis=1).numpy()
            # prediction=paddle.argmax(pre2,axis=1).numpy()
            confusionMatrix+=getConfusionMatrix(prediction,label.numpy())
            print(prediction[0].shape)
            img=Image.fromarray(prediction[0].astype('uint8'))
            
            img.save('./tmp/crfs/pre/{}.png'.format(i))
            
            # CRFs()
            
    miou,ious=getMiou(confusionMatrix)
    print('miou:{}\n{}'.format(miou,ious))

def main():
    test()
if __name__=='__main__':
    main()