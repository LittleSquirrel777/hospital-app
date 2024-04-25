#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import collections
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import sklearn.metrics as metrics

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 15,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


# import tensorflow as tf


# In[2]:


def transfom_data(train_x, train_y, batch_size=128):
    # trainx:ndarray:(80000,205) dtype:float64   trainy:ndarray:(80000,4) (onehot) dtype:int64

    # 转换数据类型
    train_x = torch.FloatTensor(train_x)  # train_x:Tensor:(8000,205):dtype:float32
    train_y = torch.FloatTensor(train_y).long()  # train_y:Tensor:(8000,4)(onehot) dtype:int64

    # 封装数据
    all_data_train = [[train_x[i], train_y[i]] for i in range(len(train_x))]
    train_loader = DataLoader(dataset=all_data_train, batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=False)
    return train_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 继承__init__功能
        ## 第一层卷积

        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=2)
        self.avepool_1 = nn.AvgPool1d(2, 2)

        self.conv_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=2)
        self.avepool_2 = nn.AvgPool1d(2, 2)

        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(in_features=192, out_features=512)
        self.fc_2 = nn.Linear(in_features=512, out_features=4)
        self.relu_3 = nn.LeakyReLU(0.02)

        ## 输出层
        self.output = nn.Softmax()

    def forward(self, x):
        x = x.reshape([-1, 1, 205])  # 128*1*205
        x = self.conv_1(x)  # 128*16*102
        x = self.avepool_1(x)  # 128*16*51
        x = self.conv_2(x)  #128*16*25
        x = self.avepool_2(x)#128*16*13

        x = x.view(x.size(0), -1)
        # x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.relu_3(x)
        # x = self.output(x)
        return x


def get_result(ture_label, pred_label):
    """
    计算评价指标
    """
    acc = metrics.accuracy_score(ture_label, pred_label)
    rec = metrics.recall_score(ture_label, pred_label, average='weighted')
    f1s = metrics.f1_score(ture_label, pred_label, average='weighted')
    pre = metrics.precision_score(ture_label, pred_label, average='weighted')

    return round(acc, 4), round(rec, 4), round(pre, 4), round(f1s, 4)


# In[3]:

if __name__ == '__main__':

    # 读取数据
    data = pd.read_csv('/home/andy/下载/train.csv', index_col=0)
    all_data = []
    for i in data.values:
        signal = i[0].split(',')
        all_data += [signal]
    all_data = pd.DataFrame(all_data)

    all_data = all_data.astype('float')  # float64
    all_label = data[['label']]  # float64
    all_label_onehot = pd.get_dummies(all_label, columns=['label'])  # ont_hot

    # #数据打包
    # train_loader=transfom_data(all_data.values[::10],
    #                            all_label_onehot.values[::10],batch_size=128)

    # 数据切分
    train_x, test_x, train_y, test_y = train_test_split(all_data, all_label_onehot,
                                                        train_size=0.8)
    # 数据打包
    train_loader = transfom_data(train_x.values,
                                 train_y.values, batch_size=128)
    test_loader = transfom_data(test_x.values,
                                test_y.values, batch_size=128)

    # 开始训练

    # In[11]:

    # 如果有gpu就用gpu，如果没有就用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = CNN().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    epochs_result = {}
    test_result = {}
    for epoch in range(20):
        epoch_loss = 0
        pred_list = []
        true_list = []

        # 模型训练
        cnn.train()
        for step, (input_x, input_y) in enumerate(train_loader):
            input_x = input_x.to(device)
            input_y = input_y.to(device)
            input_y = torch.max(input_y, 1)[1]
            size = input_x.shape[0]

            optimizer.zero_grad()
            output = cnn(input_x)

            loss = loss_function(output, input_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            pred_label = np.argmax(output.detach().to('cpu').numpy(), axis=1)
            true_label = input_y.to('cpu').tolist()

            pred_list += pred_label.tolist()
            true_list += true_label
        epochs_result[epoch] = list(get_result(true_list, pred_list)) + [epoch_loss]

        # 模型测试
        with torch.no_grad():
            cnn.eval()
            test_loss = 0
            pred_list = []
            true_list = []
            for step, (input_x, input_y) in enumerate(test_loader):
                input_x = input_x.to(device)
                input_y = input_y.to(device)
                input_y = torch.max(input_y, 1)[1]

                output = cnn(input_x)

                loss = loss_function(output, input_y)
                test_loss += loss.item()

                pred_label = np.argmax(output.detach().to('cpu').numpy(), axis=1)
                true_label = input_y.to('cpu').tolist()

                pred_list += pred_label.tolist()
                true_list += true_label

            test_result[epoch] = list(get_result(true_list, pred_list)) + [test_loss]

        print('Epoch: [%.1f]' % (epoch + 1),
              'train loss: %.4f  ' % epoch_loss,
              'test loss: %.4f   ' % test_loss,
              'test accuracy: %.4f' % (metrics.accuracy_score(true_list, pred_list)))
        torch.save(cnn.state_dict(), "./model_parameter.pkl[%.1f]"%(epoch+1))

    print(test_result)



# In[5]:


def encrypt(tensor):
    """
    模拟加密，添加一个噪声
    """
    noise = torch.randn_like(tensor)
    return tensor + noise


def decrypt(tensor):
    """
    模拟解密，减去一个噪声
    """
    noise = torch.randn_like(tensor)
    return tensor - noise


def set_secrete_model(cnn, secret_data_x):
    """
    使用同态加密算法进行对数据进行加密/解密计算，最终返回预测结果
    """
    with torch.no_grad():
        cnn.eval()
        # 开始输入数据
        x = encrypt(secret_data_x)  # 对输入数据进行【加密】
        x = secret_data_x.reshape([-1, 1, 205])

        # -------第一层卷积层--------------
        # 卷积
        x = cnn.conv_1(x)

        # 激活，需要解密，再加密
        x = decrypt(x)  # 解密
        x = cnn.relu_1(x)  # 激活
        x = encrypt(x)  # 加密

        # 均值池化
        x = cnn.avepool_1(x)

        # --------第二层卷积层-----------------
        # 卷积
        x = cnn.conv_2(x)

        # 激活，需要解密，再加密
        x = decrypt(x)  # 解密
        x = cnn.relu_2(x)  # 激活
        x = encrypt(x)  # 加密

        # 均值池化
        x = cnn.avepool_2(x)

        # --------------第一层全连接层+激活层------------
        # 全连接
        x = x.view(x.size(0), -1)
        x = cnn.fc_1(x)

        # 激活，需要解密，再加密
        x = decrypt(x)  # 解密
        x = cnn.relu_3(x)  # 激活
        x = encrypt(x)  # 加密

        # --------------第二层全连接层-----------
        x = cnn.fc_2(x)
    return x

# In[6]:


# #模拟新采集的数据
# secret_data_x_=input_x.detach().cpu().numpy()
# secret_data_y_=input_y.detach().cpu().numpy()
# secret_data_x=torch.FloatTensor(secret_data_x_).to(device)
#
# #同态加密计算
# predict=set_secrete_model(cnn,secret_data_x).to('cpu').numpy()
# secrete_pred_label=np.argmax(predict,axis=1)
# secrete_pred_label


# In[ ]:



