#  create date 2025-08-17
#  author :   chensong 
#   3、构建卷积神经网络参数 调试


## 基本环境

import torch;
import torch.nn as nn;
import torch.optim as optim;
import torch.nn.functional as F;
#  datasets  处理数据集
#  transforms 预处理操作的
from torchvision import datasets, transforms;
import matplotlib.pyplot as plt;
import numpy as np;

#%matplotlib inline


# 定义超参数

#  1.  图像的总尺寸 28 * 28
input_size = 28; 
#  2. 标签的种类数
num_classses = 10;

# 3. 训练的总循环周期
num_epochs = 3;
# 4. 一个最（批次）的大小 64张图片
batch_size = 64;

# 训练集 
train_dataset = datasets.MNIST('data/', 
                               download=True, 
                               train=True, 
                               transform=transforms.ToTensor());

# 测试集
test_dataset = datasets.MNIST('data/', 
                               download=True, 
                               train=False, 
                               transform=transforms.ToTensor());

# 构建batch数据

train_loader  = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle=True);

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle=True);



# 定义卷积网络模型

class  CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__();
        self.conv1 = nn.Sequential(          # 输入大小 （1， 28， 28）
            # 2D卷积9
            nn.Conv2d(
                in_channels = 1,             # 灰度图
                out_channels = 16,           # 要得到几多少个特征图
                kernel_size = 5,             # 卷积核大小
                stride = 1,                  # 步长
                padding = 2,                 # 如果希望卷积后大小更原来一样 需要设置padding=(kernel_size-1)/2 if stride = 1
            ),                               # 输出的特征图为（16， 28， 28）
            nn.ReLU(),                       # relu层
            nn.MaxPool2d(kernel_size = 2),   # 进行池化操作（2*2区域） 输出结果为（16， 14， 14);
            
        );

        self.conv2 = nn.Sequential(           # 下一个套餐的输入(16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),       # 输出（32， 14， 14）
            nn.ReLU(),                        # relu层
            nn.Conv2d(32, 32, 5, 1, 2),       # 
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 输出（32， 7， 7）
        );

        self.conv3 = nn.Sequential(           # 下一个套餐的输入(16, 14, 14)
            nn.Conv2d(32, 64, 5, 1, 2),       # 输出（32， 14， 14）
            nn.ReLU(),                        # 输出 （32， 7， 7）
        );


        self.out = nn.Linear(64 * 7 * 7, 10);    # 全连接层得到的结果


    # 前向传播
    def forward(self, x):
        x = self.conv1(x);
        x = self.conv2(x);
        x = self.conv3(x);
        x = x.view(x.size(0), -1);             # flatten 操作 结果为：（batch_size， 32 * 7 * 7）
        output = self.out(x); 
        return output;
        
        
        
        

# 准确率

def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1];
    # 预测值与标签是否相等的
    rights = pred.eq(labels.data.view_as(pred)).sum();
    return rights, len(labels);
    
    

# 实例化

net = CNN();

# 损失函数
criterion = nn.CrossEntropyLoss();

# 优化器    定义优化器 普通的随机梯度下降算法   lr : 学习率
optimizer  = optim.Adam(net.parameters(), lr=0.001);

# 开始训练循环
for epoch in range(num_epochs):
    # 当前epoch的结果保存下来
    train_rights = [];

    #针对容器中的每一个批进行循环
    for batch_idx, (data, target) in enumerate(train_loader): 
        net.train();
        output = net(data);
        loss = criterion(output, target);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        right = accuracy(output, target);
        train_rights.append(right);

        if batch_idx % 100 == 0:
            net.eval();
            val_rights = [];

            for (data, target) in test_loader:
                output = net(data);
                right = accuracy(output, target);
                val_rights.append(right);

            # 准确率计算
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]));
            val_r =  (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]));

            print('当前epoch：{} [{}/{} ({:.0f}%)]\t 损失: {:.6f}\t 训练集准确率:{:.2f}%\t 测试集准确率:{:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data,
                100. *train_r[0].numpy() / train_r[1],
                100. * val_r[0].numpy() / val_r[1]
            ));