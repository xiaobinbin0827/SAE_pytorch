from AE_ModelConstruction import *

percent=0.9#测试比例
patience=8#比对次数
LR=0.005
BATCH_SIZE=64
layer=3
input_num=14

ae=autoencoder(input_num,layer,batch_normalization=True)
ae2=autoencoder(input_num,layer,batch_normalization=True)

name_list=['ae_D0_temp','ae_D1_temp','ae_D2_temp']#训练数据文件名
Trainset,Trainset_max,Trainset_min=Traindata(name_list)#加载训练集
subTrainset=Trainset[0:int(percent*Trainset.shape[0])]#将训练数据分成两批，训练模型
vaildset=Trainset[int(percent*Trainset.shape[0]):-1]

torch_sTs=torch.from_numpy(subTrainset).float()
train_loader = Data.DataLoader(dataset=torch_sTs, batch_size=BATCH_SIZE, shuffle=True)
torch_vs=torch.from_numpy(vaildset).float()

#第一轮训练，将训练集分为子训练集和验证集，运行算法1并返回最佳训练轮数
i=0
j=0
Loss_min=10
i_best=i
optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
loss_func = nn.MSELoss()
while j<patience:
    i=i+1
    ae.train()  # 训练状态
    for step, x in enumerate(train_loader):  # Dataloader
        b_x = x.view(-1, 1 * 14)  # batch x, shape (batch, 1*6)
        b_y = x.view(-1, 1 * 14)  # batch y, shape (batch, 1*6)
        encoded, decoded = ae(b_x)
        loss = loss_func(decoded, b_y)  # mean square error 求误差
        optimizer.zero_grad()  # clear gradients for this training step 在这轮的step里清除梯度
        loss.backward()  # backpropagation, compute gradients 误差反向传播，计算梯度d(loss)/dx
        optimizer.step()  # apply gradients 根据梯度更新参数
    ae.eval()  # 测试状态
    vs_enc, vs_dec=ae(torch_vs)
    vs_loss=loss_func(vs_dec,torch_vs)
    print('验证集误差：',vs_loss.data.numpy())
    if vs_loss.data.numpy()<Loss_min:
        j=0
        i_best=i
        Loss_min=vs_loss.data.numpy()
    else:
        j=j+1
print('best epoch:',i_best)
#第二轮训练
train_loss=[]
optimizer2 = torch.optim.Adam(ae2.parameters(), lr=LR)
loss_func2 = nn.MSELoss()
torch_Ts=torch.from_numpy(Trainset).float()
train_loader2 = Data.DataLoader(dataset=torch_Ts, batch_size=BATCH_SIZE, shuffle=True)
ae2.train()#训练状态
for epoch in range(i_best):
    for step, x in enumerate(train_loader2):#Dataloader
        b_x = x.view(-1, 1*14)   # batch x, shape (batch, 1*14)
        b_y = x.view(-1, 1*14)   # batch y, shape (batch, 1*14)
        encoded, decoded = ae2(b_x)
        loss = loss_func2(decoded, b_y)      # mean square error
        optimizer2.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer2.step()                    # apply gradients
        train_loss.append(loss.data.numpy())
        if step % 50 == 0:
            print('Epoch: ', epoch,'| train loss: %.6f' % loss.data.numpy())
torch.save(ae2, 'ae_withBN_3.pkl')#保存神经网络模型
np.save('Loss_withBN_3.npy', train_loss)#保存训练误差