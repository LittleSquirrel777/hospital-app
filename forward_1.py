import ctypes
import os

import django
from django.test import TestCase
from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from ctypes import cdll
from hosptialapp.verifiable_cnn1 import *
os.environ.setdefault("DJANGO_SETTINGS_MODULE","hospital_2.settings")
django.setup()
from hosptialapp import models
import time

def sdigit(x):
    return round(x)

def error_location(table,layer_index,layer_name,start):
    for i in range(len(layer_index[0])):
        table.objects.filter(id=layer_index[0][i]).update(verify_res=0)
        layer_name = layer_name+"_"+str(layer_index[1])
        previous_name = table.objects.get(id=layer_index[0][i]+start).verify_error_layer
        if previous_name==None:
            table.objects.filter(id=layer_index[0][i]+start).update(verify_error_layer=layer_name)
        elif previous_name==layer_name:
            pass
        else:
            layer_name=previous_name+','+layer_name
            table.objects.filter(id=layer_index[0][i]+start).update(verify_error_layer=layer_name)


def handle_verify(start,end,res_conv1,res_pool1,res_conv2,res_pool2,res_fc2):
    print(start,end)
    medicalrecord=models.medicalrecord
    index_conv1 = np.where(res_conv1==0)
    index_conv2 = np.where(res_conv2==0)
    index_pool1 = np.where(res_pool1==0)
    index_pool2 = np.where(res_pool2==0)
    index_fc2 = np.where(res_fc2==0)
    for i in range(start,end+1):
        medicalrecord.objects.filter(id=i).update(verify_res=1)
        medicalrecord.objects.filter(id=i).update(verify_error_layer=None)
    error_location(medicalrecord,index_conv1,"conv1",start)
    error_location(medicalrecord,index_pool1,"pool1",start)
    error_location(medicalrecord,index_conv2,"conv2",start)
    error_location(medicalrecord,index_pool2,"pool2",start)
    error_location(medicalrecord,index_fc2,"fc2",start)

def verify_write(verify_res,start,layer_num):
    locations = []
    for key,value in verify_res.items():
        if value=='0':
            locations.append(key)
    medicalrecord=models.medicalrecord
    if len(locations)==0:
        return 0
    else:
        for location in locations:
            location = location.split(':')
            id = start+int(location[0])
            medicalrecord.objects.filter(id=id).update(verify_res=0)
            previous_layer_name = medicalrecord.objects.get(id=id).verify_error_layer
            if previous_layer_name==None:
                medicalrecord.objects.filter(id=id).update(verify_error_layer=layer_num)
            else:
                if previous_layer_name==layer_num:
                    pass
                else:
                    layer_name = previous_layer_name + ',' + layer_num
                    medicalrecord.objects.filter(id=id).update(verify_error_layer=layer_name)
            if layer_num!='5':
                previous_kernel_name = medicalrecord.objects.get(id=id).verify_error_kernel
                if previous_kernel_name == None:
                    if layer_num=='1' or layer_num=='3':
                        medicalrecord.objects.filter(id=id).update(verify_error_kernel=location[2])
                    else:
                        medicalrecord.objects.filter(id=id).update(verify_error_kernel=location[1])
                else:
                    if layer_num == '1' or layer_num == '3':
                        if previous_kernel_name==location[2]:
                            pass
                        else:
                            kernel_name = previous_kernel_name + ',' + location[2]
                            medicalrecord.objects.filter(id=id).update(verify_error_kernel=kernel_name)
                    else:
                        if previous_kernel_name==location[1]:
                            pass
                        else:
                            kernel_name = previous_kernel_name + ',' + location[1]
                            medicalrecord.objects.filter(id=id).update(verify_error_kernel=kernel_name)





def infer_interface(start,end):
    session = requests.session()
    torch.set_printoptions(precision=6)

    pd.set_option('max_colwidth',200)

    # if args!=None:
    #     w1_f = args[0]
    #     w2_f = args[1]
    #     w3_f = args[2]

    cnn=CNN()
    cnn.load_state_dict(torch.load('/home/www/project/hospital_2/hosptialapp/model_parameter.pkl3'))


    w1=cnn.conv_1.weight.detach().numpy()
    # b1=cnn.conv_1.bias.detach().numpy()
    sdigit_n=np.vectorize(sdigit)
    w1=sdigit_n(w1*1000000)
    #b1=sdigit_n(b1*1000000)

    w2=cnn.conv_2.weight.detach().numpy()
    #b2=cnn.conv_2.bias.detach().numpy()
    w2=sdigit_n(w2*1000000)
    #b2=sdigit_n(b2*1000000)

    # w3=cnn.fc_1.weight.detach().numpy()
    # #b3=cnn.fc_1.bias.detach().numpy()
    # w3=sdigit_n(w3*1000000)
    # #b3=sdigit_n(b3*1000000)

    w4=cnn.fc_2.weight.detach().numpy()
    #b4=cnn.fc_2.bias.detach().numpy()
    w4=sdigit_n(w4*1000000)
    #b4=sdigit_n(b4*1000000)


    cnn_numpy=Conv1d_numpy(input_channel=1, output_channel=16, kernel_size=2, stride=2)
    cnn_numpy.weight=w1
    cnn_numpy.bias=None
    pool_numpy = Avgpool_numpy(kernel_size=2,stride=2)
    cnn_numpy2 = Conv1d_numpy(input_channel=16,output_channel=16,kernel_size=2,stride=2)
    cnn_numpy2.weight=w2
    cnn_numpy2.bias=None
    # fc_1_numpy = Fc_numpy(in_channel=192,out_channel=512)
    # fc_1_numpy.weight = w3
    # fc_1_numpy.bias = None
    fc_2_numpy = Fc_numpy(in_channel=192,out_channel=4)
    fc_2_numpy.weight = w4
    fc_2_numpy.bias = None

    alter = models.alter
    try:
        alter_data = alter.objects.get(id=1)
        if alter_data != None:
            if alter_data.layer_no == '1':
                # cnn_numpy.kernel_8 = cnn_numpy.weight.copy()
                cnn_numpy.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 0] = int(alter_data.num1)
                cnn_numpy.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 1] = int(alter_data.num2)
            elif alter_data.layer_no == '3':
                cnn_numpy2.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 0] = int(alter_data.num1)
                cnn_numpy2.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 1] = int(alter_data.num2)
            elif alter_data.layer_no == '5':
                fc_2_numpy.weight[int(alter_data.output_no), int(alter_data.input_no)] = int(alter_data.weight)
    except:
        pass

    # path_enc = '/home/www/project/hospital_2/hosptialapp/encrypteds.csv'
    path_enc = '/home/data/data1.csv'
    # path_enc = '/media/andy/新加卷/data/encrypted.csv'


    coloum=list(range(617))
    data = pd.read_csv(path_enc, index_col=0,names=coloum,header=None,skiprows=lambda x:x<start or x>end)
    # data = data.iloc[start:end]
    all_data=[]
    for i in data.values:
        y0=i[0:205]
        y1=i[205:410]
        t=i[410:615]
        signal_enc=[y0,y1,t]    #一个二维列表，其中每个元素是每条数据的y0部分（一个列表）和每条数据的y1部分（一个列表）
        all_data+=signal_enc   #一个二维列表，前二个元素是第一条数据的y0部分和y1部分（每个都是列表），第3，4个元素是第二条数据的y0,y1部分，依次类推

    all_data=np.array(all_data)
    int_n=np.vectorize(int)
    all_data=int_n(all_data)  #将数据转为int类型


    all_label=data[616]
    all_label_onehot = pd.get_dummies(all_label, columns=['label'])
    all_data=all_data.reshape([-1,1,205])


    medicalrecord = models.medicalrecord
    medicalrecord.objects.update(verify_res=1)
    medicalrecord.objects.update(verify_error_layer=None)
    medicalrecord.objects.update(verify_error_kernel=None)
    time1=time.time()

    s = 0

    x,verify_conv1,s=cnn_numpy(all_data,session)


    verify_write(verify_conv1,start,'1')

    while s == 0:
        continue

    s = 0
    x,verify_pool1,s=pool_numpy(x,session)
    # print(verify_pool1)
    # print(type(verify_pool1))

    verify_write(verify_pool1,start,'2')

    while s == 0:
        continue

    s=0
    x,verify_conv2,s=cnn_numpy2(x,session)

    verify_write(verify_conv2,start,'3')

    while s == 0:
        continue
    s=0
    x,verify_pool2,s=pool_numpy(x,session)

    verify_write(verify_pool2,start,'4')


    x=x.reshape([-1,192])

    # # verify_fc1 = np.zeros([11,512], dtype='O')
    # # x,verify_fc1=fc_1_numpy(x)
    while s == 0:
        continue

    s = 0
    x,verify_fc2,s=fc_2_numpy(x,session)

    while s == 0:
        continue

    verify_write(verify_fc2,start,'5')

    time2=time.time()

    # print(start,end)
    # handle_verify(start,end,verify_conv1,verify_pool1,verify_conv2,verify_pool2,verify_fc2)
    # print(x.shape)
    msg = {}
    i = 0
    while i < x.shape[0]:
        temp = []
        temp.append(x[i].tolist())
        temp.append(x[i + 1].tolist())
        temp.append(x[i + 2].tolist())
        msg.update({'l' + str(int(i / 3)): temp})
        i += 3
    # print(msg)
    print('推理数据条数：',len(msg))
    print('起点终点',start, end)
    return msg





if __name__=='__main__':
    session = requests.session()
    torch.set_printoptions(precision=6)

    pd.set_option('max_colwidth',200)
    cnn=CNN()
    cnn.load_state_dict(torch.load('./model_parameter.pkl3'))

    w1=cnn.conv_1.weight.detach().numpy()
    # b1=cnn.conv_1.bias.detach().numpy()
    sdigit_n=np.vectorize(sdigit)
    w1=sdigit_n(w1*1000000)
    # b1=sdigit_n(b1*1000000)

    w2=cnn.conv_2.weight.detach().numpy()
    # b2=cnn.conv_2.bias.detach().numpy()
    w2=sdigit_n(w2*1000000)
    # b2=sdigit_n(b2*1000000)

    # w3=cnn.fc_1.weight.detach().numpy()
    # # b3=cnn.fc_1.bias.detach().numpy()
    # w3=sdigit_n(w3*1000000)
    # b3=sdigit_n(b3*1000000)

    w4=cnn.fc_2.weight.detach().numpy()
    # b4=cnn.fc_2.bias.detach().numpy()
    w4=sdigit_n(w4*1000000)
    # b4=sdigit_n(b4*1000000)




    cnn_numpy=Conv1d_numpy(input_channel=1, output_channel=16, kernel_size=2, stride=2)
    cnn_numpy.weight=w1
    cnn_numpy.bias=None
    pool_numpy = Avgpool_numpy(kernel_size=2,stride=2)
    cnn_numpy2 = Conv1d_numpy(input_channel=16,output_channel=16,kernel_size=2,stride=2)
    cnn_numpy2.weight=w2
    cnn_numpy2.bias=None
    # fc_1_numpy = Fc_numpy(in_channel=192,out_channel=512)
    # fc_1_numpy.weight = w3
    # fc_1_numpy.bias = None
    fc_2_numpy = Fc_numpy(in_channel=192,out_channel=4)
    fc_2_numpy.weight = w4
    fc_2_numpy.bias = None

    alter = models.alter
    try:
        alter_data = alter.objects.get(id=1)
        if alter_data != None:
            if alter_data.layer_no == '1':
                # cnn_numpy.kernel_8 = cnn_numpy.weight.copy()
                cnn_numpy.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 0] = int(alter_data.num1)
                cnn_numpy.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 1] = int(alter_data.num2)
            elif alter_data.layer_no == '3':
                cnn_numpy2.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 0] = int(alter_data.num1)
                cnn_numpy2.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 1] = int(alter_data.num2)
            elif alter_data.layer_no == '5':
                fc_2_numpy.weight[int(alter_data.output_no), int(alter_data.input_no)] = int(alter_data.weight)
    except:
        pass


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # path_enc = '/home/www/project/hospital_2/hosptialapp/encrypteds.csv'
    path_enc = '/home/data/data1.csv'
    # path_enc = '/media/andy/新加卷/data/encrypted.csv'
    # path_plain = '/home/andy/下载/train.csv'


    # data_p = pd.read_csv(path_plain, index_col=0,skiprows=lambda x:x<1 or x>11)
    # all_data_p = []
    # for i in data_p.values:
    #     signal = i[0].split(',')
    #     all_data_p += [signal]
    # all_data_p = pd.DataFrame(all_data_p)
    # all_data_p = all_data_p.astype('float')  # float64
    # train_x = torch.FloatTensor(all_data_p.values)
    # train_x = train_x.to(device)
    # train_x = train_x.reshape([-1, 1, 205])  # 128*1*205
    # train_x = cnn.conv_1(train_x)  # 128*16*102
    # train_x = cnn.avepool_1(train_x)  # 128*16*51
    # train_x = cnn.conv_2(train_x)  # 128*16*25
    # train_x = cnn.avepool_2(train_x)  # 128*16*13
    #
    # train_x = train_x.view(train_x.size(0), -1)
    # # x = self.flatten(x)
    # train_x = cnn.fc_1(train_x)
    # x_p = cnn.fc_2(train_x)

    coloum=list(range(617))

    start = 30026
    end = 30026
    data = pd.read_csv(path_enc, index_col=0,names=coloum,header=None,skiprows=lambda x:x<start or x>end)
    # data = data.iloc[0:20]
    all_data=[]
    for i in data.values:
        y0=i[0:205]
        y1=i[205:410]
        t=i[410:615]
        signal_enc=[y0,y1,t]    #一个二维列表，其中每个元素是每条数据的y0部分（一个列表）和每条数据的y1部分（一个列表）
        all_data+=signal_enc   #一个二维列表，前二个元素是第一条数据的y0部分和y1部分（每个都是列表），第3，4个元素是第二条数据的y0,y1部分，依次类推

    all_data=np.array(all_data)
    int_n=np.vectorize(int)
    all_data=int_n(all_data)  #将数据转为int类型
    # type_a=all_data.dtype
    # res=all_data*all_data
    # type_r=res.dtype
    # shape=res.shape


    #all_data=pd.DataFrame(all_data)  #二维数据，前二个元素是第一条数据的y0部分和y1部分（每个都是series），第3，4个元素是第二条数据的y0,y1部分，依次类推
    #all_data1=all_data.astype('float')
    all_label=data[616]
    all_label_onehot = pd.get_dummies(all_label, columns=['label'])
    all_data=all_data.reshape([-1,1,205])

    medicalrecord = models.medicalrecord
    medicalrecord.objects.update(verify_res=1)
    medicalrecord.objects.update(verify_error_layer=None)
    medicalrecord.objects.update(verify_error_kernel=None)
    time1=time.time()

    s = 0

    x,verify_conv1,s=cnn_numpy(all_data,session)

    verify_write(verify_conv1,start,'1')

    while s == 0:
        continue

    s = 0
    x,verify_pool1,s=pool_numpy(x,session)

    verify_write(verify_pool1,start,'2')

    while s == 0:
        continue

    s=0
    x,verify_conv2,s=cnn_numpy2(x,session)

    verify_write(verify_conv2,start,'3')

    while s == 0:
        continue
    s=0
    x,verify_pool2,s=pool_numpy(x,session)

    verify_write(verify_pool2,start,'4')


    x=x.reshape([-1,192])

    # # verify_fc1 = np.zeros([11,512], dtype='O')
    # # x,verify_fc1=fc_1_numpy(x)
    while s == 0:
        continue
    x,verify_fc2,s=fc_2_numpy(x,session)

    verify_write(verify_fc2,start,'5')

    time2=time.time()
    # handle_verify(start,end,verify_conv1,verify_pool1,verify_conv2,verify_pool2,verify_fc2)


    # ont_hot DataFrame:(100000,4)
    #test_x = torch.FloatTensor(all_data.values)    #二维Tensor，前二个元素是第一条数据的y0部分和y1部分（每个都是tensor），第3，4个元素是第二条数据的y0,y1部分，依次类推

    #test_y = torch.FloatTensor(all_label_onehot.values).long()
    #test_x1 = test_x.to(device)
    #test_x1 = test_x1.reshape([-1, 1, 205])  #三维数据，前二个元素是第一条数据的y0部分和y1部分（每个都是二维tensor(经过reshape,每个部分都有1一个通道)），第3，4个元素是第二条数据的y0,y1部分，依次类推
    #test_x2 = cnn.conv_1(test_x1)
    #test_x1_np = np.array(test_x1)
    #test_x2_np = cnn_numpy(test_x1_np)
    #test22=test_x2.detach().numpy()
    # print(str(time2-time1))
    pass
