# SAE_pytorch
## 基于pytorch实现的堆叠自编码神经网络，包含网络模型构造、训练、测试
主要包含训练与测试数据（.mat文件）、模型（AE_ModelConstruction.py、AE_Train.py）以及`测试例子`（AE_Test.py）<br>
其中`ae_D_temp`为训练数据，`ae_Kobs3_temp`为正常测试数据，`ae_ver_temp`为磨煤机堵煤故障数据，数据集包含风粉混合物温度等14个变量<br>
### `AE_ModelConstruction.py`文件：
```
包含两部分，1是构造神经网络类autoencoder()，2是训练数据的导入方法Traindata()
在程序中神经网络的层数和每层神经元个数没有固定，可根据使用者的输入值来构造神经网络，方便调试
autoencoder类在初始化时有三个参数，第一个是网络输入值，第二个是SAE编码过程的层数(编码、解码过程层数相同)，第三个是是否添加BN层
这里为了构造方便，给每层神经元的个数与层数建立一个关系：第一层神经元的个数为2^(layer数+2)，之后逐层为上一层的1/2
```
### `AE_Train.py`文件：
```
该文件为模型训练的程序，依赖于AE_ModelConstruction.py。训练过程包含两部分：
第一个是记录最小训练误差的那一轮次，得到训练所需的epoch
第二个是正常的模型训练，训练完毕后保存模型参数与训练误差
保存文件命名规则：withBN代表包含BN层，_layer代表网络层数
```
### `AE_Test.py`文件：
```
在AE_Train.py中可以训练多个神经网络模型，那么在测试时可以在layer_index添加你想对比效果的神经网络
比如在训练中保存了ae_withBN_2.pkl、ae_withBN_3.pkl、ae_withBN_4.pkl三个模型，那么令layer_index=[2,3,4]
即可同时将各个模型的测试效果同时展示(各变量最大相对误差、平均相对误差、各变量估计偏差计算得到的相似度)
```
