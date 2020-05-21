# SAE_pytorch
## 基于pytorch实现的堆叠自编码神经网络，包含网络模型构造、训练、测试
主要包含训练与测试数据（.mat文件）、模型（AE_ModelConstruction.py、AE_train.py）以及`测试例子`（example.py）<br>
其中`ae_D_temp`为训练数据，`ae_Kobs3_temp`为正常测试数据，`ae_ver_temp`为磨煤机堵煤故障数据，数据集包含风粉混合物温度等14个变量<br>
### `AE_ModelConstruction.py`文件：
```
包含两部分，1是网络模型的构造(`class autoencoder()`)，2是训练数据的导入方法(`def Traindata()`)<br>
在程序中神经网络的层数和每层神经元个数没有固定，可根据使用者的输入值来构造神经网络，方便调试。`autoencoder`类在初始化时有三个参数，
第一个是网络输入值，第二个是SAE编码过程的层数(编码、解码过程层数相同)，第三个是是否添加BN层。这里为了构造方便，给每层神经元的个数与
层数建立一个关系：第一层神经元的个数为2^(layer数+2)，之后逐层为上一层的1/2<br>
```
