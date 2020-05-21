import torch
from torch import nn
from scipy import io
import numpy as np
import torch.utils.data as Data

ACTIVATION = torch.nn.LeakyReLU()

class autoencoder(nn.Module):
    def __init__(self,input_num,layer,batch_normalization=False):
        super(autoencoder,self).__init__()
        self.do_bn=batch_normalization
        self.encoder=nn.Sequential()
        for i in range(layer):
            # encoder第一层
            if i == 0:
                self.encoder.add_module('encoder_linear%i' % (i + 1), nn.Linear(input_num, 2 ** (layer + 2)))
                if self.do_bn:
                    self.encoder.add_module('encoder_BN%i' % (i + 1),
                                        nn.BatchNorm1d(2 ** (layer + 2), momentum=0.5))
                self.encoder.add_module('encoder_actFun%i' % (i + 1), ACTIVATION)
            # encoder最后一层
            elif i == (layer - 1):
                self.encoder.add_module('encoder_linear%i' % (i + 1), nn.Linear(16, 3))
                if self.do_bn:
                    self.encoder.add_module('encoder_BN%i' % (i + 1),nn.BatchNorm1d(3,momentum=0.5))
            # encoder其它层
            else:
                self.encoder.add_module('encoder_linear%i'%(i+1),nn.Linear(2 ** (layer - i + 3),2 **(layer-i+2)))
                if self.do_bn:
                    self.encoder.add_module('encoder_BN%i' % (i + 1),
                                        nn.BatchNorm1d(2 ** (layer - i + 2), momentum=0.5))
                self.encoder.add_module('encoder_actFun%i' % (i + 1), ACTIVATION)
        self.decoder = nn.Sequential()
        for i in range(layer):
            # decoder第一层
            if i == 0:
                self.decoder.add_module('decoder_linear%i' % (i + 1), nn.Linear(3, 16))
                if self.do_bn:
                    self.decoder.add_module('decoder_BN%i' % (i + 1), nn.BatchNorm1d(16, momentum=0.5))
                self.decoder.add_module('decoder_actFun%i' % (i + 1), ACTIVATION)
            #decoder最后一层
            elif i == (layer - 1):
                self.decoder.add_module('decoder_linear%i' % (i + 1), nn.Linear(2 ** (layer+ 2), input_num))
                self.decoder.add_module('decoder_actFun%i' % (i + 1), nn.Sigmoid())
            # 其它层
            else:
                self.decoder.add_module('encoder_linear%i' % (i + 1), nn.Linear(2 ** (3+i),2 ** (4+i)))
                if self.do_bn:
                    self.decoder.add_module('encoder_BN%i' % (i + 1),nn.BatchNorm1d(2 ** (4+i), momentum=0.5))
                self.decoder.add_module('encoder_actFun%i' % (i + 1), ACTIVATION)
    def forward(self, x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return encoded,decoded
#导入训练集并归一化,训练集可以为包含多个文件的列表
def Traindata(name_list,if_nor=True):
    #name_list:文件名字列表；if_nor:是否归一化
    column_num = 18 #文件中总列数为18，但是前四列没有用到，后面会删除。
    np_D = np.zeros((1, column_num))#构造一个1行18列的全零向量
    for i in range(len(name_list)):#循环读取各个文件数据
        dict_obj = io.loadmat(name_list[i])
        temp = dict_obj['ae_D']
        np_D = np.vstack((np_D, temp))
    np_D = np.delete(np_D, 0, axis=0)
    np_D = np_D[:,4:]#去掉不需要的前四列
    index = np.where(np_D[:,3]< 10)[0]#将磨煤机电流低于10的值删去
    np_D=np.delete(np_D,index,axis=0)#删去开始构造的全零向量
    np_Dmax, np_Dmin = np_D.max(axis=0), np_D.min(axis=0)
    if if_nor:
        np_D = (np_D - np_Dmin) / (np_Dmax - np_Dmin)
        print('已归一化的训练集，大小为：', np_D.shape)
        return np_D,np_Dmax,np_Dmin
    else:
        print('未归一化的训练集，大小为：', np_D.shape)
        return np_D, np_Dmax, np_Dmin
if __name__=="__main__":
    ae=autoencoder(14,7,batch_normalization=False)
    print(ae)