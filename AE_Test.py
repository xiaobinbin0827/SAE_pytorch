import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from AE_ModelConstruction import *
BATCH_SIZE2 = 60
threshold=0.7845413302380952
layer=2
input_num=14

name_list=['ae_D0_temp','ae_D1_temp','ae_D2_temp']#数据文件名
np_D,np_Dmax,np_Dmin=Traindata(name_list)#加载训练集

#选择测试集还是验证集
# dict_Kobs=io.loadmat('ae_Kobs3_temp') #测试集
# np_Kobs=dict_Kobs['ae_Kobs2']
dict_Kobs=io.loadmat('ae_ver_temp')   #验证集
np_Kobs=dict_Kobs['ae_ver_temp']

np_Kobs=np_Kobs[:,4:]
np_Kobs=(np_Kobs-np_Dmin)/(np_Dmax-np_Dmin)
Kobs_num=np.size(np_Kobs,axis=0)

torch_Kobs=torch.from_numpy(np_Kobs).float()
print(torch_Kobs.shape)
test_loader = Data.DataLoader(dataset=torch_Kobs, batch_size=BATCH_SIZE2, shuffle=False)

layer_index=[2,3]
maxerror=[]
meanerror=[]
sims=[]
for step,layer in enumerate(layer_index):
    ae_test=autoencoder(input_num,layer,batch_normalization=True)
    namestr = 'ae_withBN_' + '%d' % layer + '.pkl'#神经网络命名规则：withBN代表包含BN层，_layer代表网络层数
    ae_test = torch.load(namestr)#加载神经网络
    test_Kobs=np.zeros((1,14))
    ae_test.eval()#测试状态
    for step,s in enumerate(test_loader):
        test_enc,test_dec=ae_test(s)
        #反归一化
        np_dec=test_dec.data.numpy()
        np_dec=np_dec*(np_Dmax-np_Dmin)+np_Dmin
        test_Kobs=np.vstack((test_Kobs,np_dec))
    test_Kobs=np.delete(test_Kobs,0,axis=0)
    np_Kobs = np_Kobs * (np_Dmax - np_Dmin) + np_Dmin#反归一化
    #各变量图片
    plt.rcParams['font.sans-serif']=['SimHei']#图片显示中文
    plt.rcParams['axes.unicode_minus'] = False
    label=['风粉混合物温度/℃','反作用力加载油压/MPa',
           '加载油压/MPa','磨煤机电流/A','一次风压力/kPa','密封风母管压力/kPa','一次风与密封风差压/kPa',
           '出入口差压/kPa','油箱油温/℃','一次风流量/t·h-1','轴承温度/℃','推力瓦温/℃','油池油温/℃','实际功率/MW']
    e=np.ones((Kobs_num,14))
    maxe=np.ones((1,14))
    meane=np.ones((1,14))
    for j in range(14):
        plt.subplot(211)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        plt.plot(np_Kobs[:, j],'steelblue',label='观测值',lw=1.5)
        plt.plot(test_Kobs[:, j],'indianred',label='重构值',lw=1.5)
        plt.legend(loc='upper right',fontsize=13)
        plt.xlabel('样本序号',fontsize=20)
        plt.ylabel(label[j],fontsize=20,verticalalignment='bottom')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplot(212)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        e[:,j]=((np_Kobs[:, j] - test_Kobs[:, j]) / np_Kobs[:, j])*100
        maxe[:,j]=np.max(abs(e[:,j]))
        meane[:,j] = np.mean(abs(e[:, j]))
        plt.plot(e[:,j],'peru',lw=1)#偏离度
        plt.xlabel('样本序号', fontsize=20)
        plt.ylabel('相对误差/%',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=18)
        plt.show()
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    maxerror.append(maxe)
    meanerror.append(meane)
    #观测值与估计值的欧式距离
    np_Kobs = (np_Kobs - np_Dmin) / (np_Dmax - np_Dmin)
    test_Kobs = (test_Kobs - np_Dmin) / (np_Dmax - np_Dmin)
    dist_norm=[]
    for i in range(Kobs_num):
        dist_norm.append(np.linalg.norm( np_Kobs[i,:] - test_Kobs[i,:] ))
    dist_norm_arr=np.array(dist_norm,dtype=float)#欧式距离
    #观测值与估计值的余弦距离
    dist_cos=[]
    for i in range(Kobs_num):
           dist_cos.append(np.dot(np_Kobs[i, :], test_Kobs[i, :]) /
                (np.linalg.norm(np_Kobs[i, :]) * np.linalg.norm(
                            test_Kobs[i, :] )))#dot向量内积，norm向量二范数
    dist_cos_arr=(np.array(dist_cos, dtype=float) * 0.5 + 0.5)#余弦距离
    #阈值
    sim=(1/(1+dist_norm_arr/dist_cos_arr))#相似度
    sims.append(sim)
    # threshold=np.min(sim)*0.98  #验证时注释
    # print('threshold=',threshold)

#将不同层数的神经网络模型的估计效果在同一个图中展示：
for step,layer in enumerate(layer_index):
    plt.plot(maxerror[step].reshape((14,)),lw=0.8,label='ae_withBN_' + '%d' % layer)
plt.legend(loc='upper left', fontsize=12)
plt.xlabel('变量序号', fontsize=18)
plt.ylabel('最大相对误差', fontsize=18)
plt.show()
for step,layer in enumerate(layer_index):
    plt.plot(meanerror[step].reshape((14,)),lw=0.8,label='ae_withBN_' + '%d' % layer)
plt.legend(loc='upper left', fontsize=12)
plt.xlabel('变量序号', fontsize=18)
plt.ylabel('平均相对误差', fontsize=18)
plt.show()
for step,layer in enumerate(layer_index):
    plt.plot(sims[step].reshape((Kobs_num,)), lw=0.8, label='ae_withBN_' + '%d' % layer)
plt.legend(loc='lower left', fontsize=12)
plt.xlabel('样本序号',fontsize=18)
plt.ylabel('相似度',fontsize=18)
plt.ylim((0, 1))
plt.show()