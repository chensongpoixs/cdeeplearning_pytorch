#  create date 2025-08-14
#  author :   chensong 
#  分类demo 

import torch;

#print(torch.__version__);




from pathlib  import Path;

import requests;

DATA_PATH = Path('data');

# 下载字体数据集目录的位置
PATH = DATA_PATH / 'mnist';

PATH.mkdir(parents=True, exist_ok=True);

URL = 'https://resources.oreilly.com/live-training/inside-unsupervised-learning/-/raw/master/data/mnist_data/';

FILENAME = 'mnist.pkl.gz';


if not (PATH / FILENAME).exists():
   content = requests.get(URL + FILENAME).content;
   (PATH / FILENAME).open('wb').write(content);
   



import pickle;
import gzip;


from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib import pyplot;
#import torch.utils.data as tud;
from torch.utils.data import TensorDataset;
from torch.utils.data import DataLoader;


USE_CUDA = torch.cuda.is_available();
import pickle
import gzip
# 打开下载的pkl.gz文件
with gzip.open((PATH / FILENAME).as_posix(), 'rb') as f:
    #mnist_data = pickle.load(f, encoding='latin1');
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1');
# mnist_data现在是一个包含两个元素的元组
# 第一个元素是训练数据集，第二个元素是测试数据集
#train_data, test_data = mnist_data
# 训练数据集包含70000个样本和784个特征
# 测试数据集包含10000个样本和784个特征

pyplot.imshow(x_train[0].reshape(28, 28), cmap='gray')
print(x_train.shape);
# 下载MNIST数据集
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = datasets.MNIST('data/MNIST_data/', download=True, train=True, transform=transform)
# testset = datasets.MNIST('data/MNIST_data/', download=True, train=False, transform=transform)

# # 可视化数据集图像
# n = 10  # 展示10张图像
# plt.figure(figsize=(10, 5))
# for i in range(n):
#     images, labels = trainset[i]
   
#     #((x_train, y_train), (x_valid, y_valid), _) = trainset[i];
#    # print(trainset[i][:10]);
#    # print('===================image=================================\n');
#    # print(images);
#    # print('===================labels=================================\n');
#    # print(labels);
#    # print('====================================================\n');
#     plt.subplot(2, 5, i+1)
#     plt.imshow(images[0].view(28, 28), cmap='gray')
#     plt.title(f'Label: {labels}')
# plt.show()
 
# x_train, _ = trainset[0];
# y_train, _ = trainset[1];
# x_valid, _ = trainset[2];
# y_valid, _ = trainset[3];



x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid ));


#n, c = x_train.shape; 
x_train, x_train.shape, y_train.min(), y_train.max();
print(x_train, y_train);
print('------------x_train.shap---------------');
print(x_train.shape);
print(y_train.min(), y_train.max());





# torch.nn.functional  很多层和函数在这里都会见到

import torch.nn.functional  as F;

loss_func = F.cross_entropy;


# 权重 
#weights = torch.randn([784, 10],  dtype = torch.float, requests_grad = True);

weights = torch.randn([784, 10]);


#bias = torch.zeros(10, requests_grad=True);

bias = torch.zeros(10);

def model(xb):
    return xb.mm(weights) + bias;


bs = 64;

xb = x_train[0:bs]; # a mini-batch from x 
yb = y_train[0:bs];



print('===============loss_func ==============================');
#print(loss_func(model(xb), yb));
print('=======================================================');
# torch  => Gpu => tensor

# numpy => cpu => array



print('-------------------------定义神经网络二层结构tarn-------------------------------------------');
#  创建一个model来更简化代码
#      1. 必须继承nn.Module 且在构造函数中需调用nn.Module的构造函数
#      2. 无需写反向传播函数， nn.Moudle能够利用autograd自动实现反向传播
#      3. Module中的可学习参数可以通过named_parameters()或者parameters()返回迭代器
from torch import nn ;

class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        #  全链接一层    输入 784特征值  输出 128特征值
        self.hidden1 = nn.Linear(784, 128);
        #  全链接二层   输入 128  输出 256
        self.hidden2 = nn.Linear(128, 256);
        
        # 分类输出 10
        self.out = nn.Linear(256, 10);
        #  暴裂因子
        self.dropout = nn.Dropout(0.5);
    # 1. 前向传播  => 需要我们自己定义怎么走动   
    # 2. 反向传播  => 是自动哈 不需要我们定义怎么走动啦~~~
    def forward(self, x):
        x = F.relu(self.hidden1(x));
        x = self.dropout(x);
        x = F.relu(self.hidden2(x));
        x = self.dropout(x);
        x = self.out(x);
        return x;




# 输出结构

net =  Mnist_NN();
print(net);

print('---------------------------------------');


for name, parameter in net.named_parameters():
    print(name, parameter, parameter.size());

 
#print(trainset);
#((x_train, y_train), (x_valid, y_valid), _) = trainset;



print(y_train[:10]);

#   28, 64,  256 


print(bs);


#  2. 使用TensorDataset 和 DataLoader来简化




train_ds = TensorDataset(x_train, y_train);
# DataLoader 作用是从数据中打包train_ds向 CPU或者GPU送数据包一次的大小
#        |      data    |
#  mem   | train_ds = 64 | 64 | 64 |
#            
# GPU/CPU|   train_ds =64 |   
#    batch_size 一批多少个包 
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True);


valid_ds = TensorDataset(x_valid, y_valid);
valid_dl = DataLoader(valid_ds, batch_size=bs*2);


def get_data(train_ds, valid_ds, bs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(valid_ds, batch_size=bs * 2));



#  一般在train 模型时加上model.train(), 这样会正常使用Batch Normalization和Dropout
#  测试的时候一般选择model.eval(), 这样就不会使用 Batch Normlization 和Dropout




import numpy as np;
import torch.nn.functional  as F;

loss_func = F.cross_entropy;

############## TRAN  ##########################
def fit(setps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(setps):
        #  训练模式 需要每次更新权重参数weight的
        model.train();
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt);
        # 验证模式  不更新权重参数
        model.eval();
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            );
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums);
        print('当前step: ' + str(step), '验证集损失:' + str(val_loss));
        
        
        




# test

a = [1, 2, 3];
b = [4, 5, 6];
zipped = zip(a, b);

print(list(zipped));

a2, b2 = zip(*zip(a, b));
print(a2);
print(b2);





from torch import optim;

def get_model():
    model = Mnist_NN();
    return model, optim.SGD(model.parameters(), lr=0.001); # Adam
    
    
    
  




def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb);

    if opt is not None:
        loss.backward();
        opt.step();
        opt.zero_grad();
        
    return loss.item(), len(xb);




train_dl, valid_dl = get_data(train_ds, valid_ds, bs);
model, opt = get_model();
fit(10, model, loss_func, opt, train_dl, valid_dl);

    
