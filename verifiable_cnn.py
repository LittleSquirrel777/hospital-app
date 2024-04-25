import ctypes

from django.test import TestCase
from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE","hospital_2.settings")
django.setup()

import torch

from ctypes import cdll

class cipher(ctypes.Structure):
    pass

cipher._fields_=[
    ('y0',ctypes.c_char*200),
    ('y1',ctypes.c_char*200),
    ('t',ctypes.c_char*200)
]


def int_charp(x):
    x=str(x)
    x=x.encode()
    return x

def package(y0,y1,t,c):
    for i in range(len(y0)):
        c[i]=cipher(y0[i],y1[i],t[i])
    return c


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 继承__init__功能
        ## 第一层卷积

        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=2,bias=False)
        self.relu_1 = nn.ReLU()
        self.avepool_1 = nn.AvgPool1d(2, 2)

        self.conv_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=2,bias=False)
        self.relu_2 = nn.ReLU()
        self.avepool_2 = nn.AvgPool1d(2, 2)

        self.flatten = nn.Flatten()
        # self.fc_1 = nn.Linear(in_features=192, out_features=512,bias=False)
        self.relu_3 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=192, out_features=4,bias=False)

        ## 输出层
        self.output = nn.Softmax()

    def forward(self, x):
        x = x.reshape([-1, 1, 205])  # 128*1*205
        x = self.conv_1(x)  # 128*16*102
        x = self.relu_1(x)
        x = self.avepool_1(x)  # 128*16*51
        x = self.conv_2(x)  #
        x = self.relu_2(x)
        x = self.avepool_2(x)

        x = x.view(x.size(0), -1)
        # x = self.flatten(x)
        x = self.fc_1(x)
        x = self.relu_3(x)
        x = self.fc_2(x)
        # x = self.output(x)
        return x





class Conv1d_numpy:
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.weight = np.random.randn(output_channel, input_channel, self.kernel_size)
        self.weight_ = np.random.randn(output_channel, input_channel, self.kernel_size)
        self.bias = True
        self.flag = False
        if bias:
            self.bias = np.random.randn(output_channel)

    def __call__(self, inputs):
        return self.infer(inputs)

    def infer(self, inputs):
        # 根据参数，算出输出的shape
        batch_size, input_channel, width = inputs.shape
        output_w = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        outputs = np.zeros([batch_size, self.output_channel, output_w],dtype='O')
        # coefficient = np.zeros([batch_size, self.output_channel, output_w],dtype='O')

        # 计算padding之后的inputs_array
        inputs_padding = np.zeros([batch_size, input_channel,  width + 2 * self.padding],dtype='O')
        inputs_padding[:, :, self.padding:self.padding + width] = inputs    #两边填充padding，将输入放在中间



        # 如果有dilation，根据dilation之后的shape往kernel中插入0（注意，原self.weight不变）
        dilation_shape = self.dilation * (self.kernel_size - 1) + 1   #dilation_shape=2
        #dilation_shape = self.dilation[0] * (self.kernel_size[0] - 1) + 1, self.dilation[1] * (self.kernel_size[1] - 1) + 1
        kernel = np.zeros((self.output_channel, input_channel, dilation_shape))
        kernel_f = np.zeros((self.output_channel, input_channel, dilation_shape))

        if self.dilation > 1:
            for i in range(self.kernel_size[0]):
                for j in range(self.kernel_size[1]):
                    kernel[:, :, self.dilation[0] * i, self.dilation[1] * j] = self.weight[:, :, i, j]
        else:
            kernel = self.weight
        if self.flag==True:
            if self.dilation > 1:
                for i in range(self.kernel_size[0]):
                    for j in range(self.kernel_size[1]):
                        kernel_f[:, :, self.dilation[0] * i, self.dilation[1] * j] = self.weight_[:, :, i, j]
            else:
                kernel_f= self.weight_

        batch_size1 = int(batch_size / 3)
        c_array2 = ctypes.c_char_p * (input_channel * dilation_shape)
        c_array1 = c_array2 * (batch_size1 * self.output_channel)
        c_array3 = ctypes.c_long * (input_channel * dilation_shape)
        c_array0 = c_array3 * (batch_size1 * self.output_channel)

        verify_res = np.zeros([batch_size1, self.output_channel, output_w],dtype='O')

        # 开始前向计算
        for w in range(output_w):   #不包括output_w，相当于input_w为奇数的话忽略最后一列
            input_ = inputs_padding[
                     :,
                     :,
                     w * self.stride:w * self.stride + dilation_shape
                     ]

            # input_ shape : batch_size, output_channel, input_channel, dilation_shape
            input_ = np.repeat(input_[:, np.newaxis, :, :], self.output_channel, axis=1)

            # kernel_ shape: batch_size, output_channel, input_channel, dilation_shape
            kernel_ = np.repeat(kernel[np.newaxis, :, :, :], batch_size, axis=0)

            if self.flag==True:
                kernel__ = np.repeat(kernel_f[np.newaxis, :, :, :], batch_size, axis=0)

                # output shape: batch_size, output_channel
                output = input_ * kernel__
                output = np.sum(output, axis=(-1, -2))
                outputs[:,:,w] = output
            else:

                # output shape: batch_size, output_channel
                output = input_ * kernel_
                output = np.sum(output, axis=(-1, -2))
                outputs[:, :, w] = output


            #verify

            input_t = input_[2::3,:,:,:].reshape(batch_size1,self.output_channel,-1)
            input_t = input_t.reshape(batch_size1*self.output_channel,-1)
            d = input_t[0]
            d1=input_t[1]
            encode = np.vectorize(int_charp)
            input_t = encode(input_t)
            t = c_array1()
            for i in range(len(t)):
                temp = c_array2()
                temp[:] = input_t[i]
                t[i] = temp

            coefficient = kernel_[2::3,:,:,:].reshape(batch_size1,self.output_channel,-1)
            coefficient = coefficient.reshape(batch_size1*self.output_channel,-1)
            h=coefficient[0]
            h1=coefficient[1]
            co = c_array0()
            for i in range(len(co)):
                temp = c_array3()
                temp[:] = coefficient[i]
                co[i] = temp

            output_y0 = output[0::3,:]
            output_y0 = output_y0.reshape(batch_size1*self.output_channel)
            output_y0 = encode(output_y0)

            output_y1 = output[1::3,:]
            output_y1 = output_y1.reshape(batch_size1*self.output_channel)
            output_y1 = encode(output_y1)

            output_t = output[2::3,:]
            output_t = output_t.reshape(batch_size1*self.output_channel)
            j=d*h
            j1=d1*h1
            r=j.sum()
            r1=j1.sum()
            output_t = encode(output_t)

            cipher_array = cipher*(batch_size1*self.output_channel)
            output_c = cipher_array()
            output_c = package(output_y0,output_y1,output_t,output_c)


            c_path='/home/andy/下载/libset10.so'
            so_lib=cdll.LoadLibrary(c_path)
            res_array = ctypes.c_bool*len(output_c)
            res = res_array()
            res_n=np.zeros([batch_size1*self.output_channel],dtype='O')
            # res = so_lib.verifyApi(output_cc[1],t[1],2,co[1],2)
            if input_channel==1:
                so_lib.verifyApiBatch(res,output_c,len(output_c),t,co)
            else:
                so_lib.verifyApiBatchFC(res,output_c,len(output_c),t,32,co,32)
            # so_lib.verifyApiBatch(res, output_c, len(output_c), t,co)
            res_n[:] = res
            res_n=res_n.reshape(batch_size1,self.output_channel)
            verify_res[:,:,w]=res_n


        if self.bias is not None:
            bias_ = np.tile(self.bias.reshape(-1, 1), (1, output_w)). \
                reshape(self.output_channel, output_w)
            outputs += bias_
        return (outputs,verify_res)

class Avgpool_numpy:
    def __init__(self,kernel_size,stride,padding=0,ceil_mode=False, count_include_pad=True):
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.ceil_mode=ceil_mode
        self.count_include_pad=count_include_pad

    def __call__(self, inputs):
        return self.infer(inputs)

    def infer(self,inputs):
        # 根据参数，算出输出的shape
        batch_size, input_channel, width = inputs.shape
        output_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        outputs = np.zeros([batch_size, input_channel, output_w], dtype='O')

        # 计算padding之后的inputs_array
        inputs_padding = np.zeros([batch_size, input_channel,  width + 2 * self.padding],dtype='O')
        inputs_padding[:, :, self.padding:self.padding + width] = inputs    #两边填充padding，将输入放在中间
        batch_size1 = int(batch_size / 3)
        verify_res = np.zeros([batch_size1,input_channel,output_w], dtype='O')

        # 开始前向计算
        for w in range(output_w):   #不包括output_w，相当于input_w为奇数的话忽略最后一列

            # input_ shape : batch_size, input_channel, kernel_size
            input_ = inputs_padding[
                     :,
                     :,
                     w * self.stride:w * self.stride + self.kernel_size
                     ]

            output = np.sum(input_, axis=(-1))*5

            outputs[:,:,w] = output

            #verify
            c_array2 = ctypes.c_char_p * (self.kernel_size)
            c_array1 = c_array2 * (batch_size1 * input_channel)
            c_array3 = ctypes.c_long * (self.kernel_size)
            c_array0 = c_array3 * (batch_size1 * input_channel)

            input_t = input_[2::3,:,:]
            input_t = input_t.reshape(batch_size1*input_channel,-1)
            encode = np.vectorize(int_charp)
            input_t = encode(input_t)
            t = c_array1()
            for i in range(len(t)):
                temp = c_array2()
                temp[:] = input_t[i]
                t[i] = temp

            co = c_array0()
            for i in range(len(co)):
                temp = c_array3(5,5)
                co[i] = temp

            output_y0 = output[0::3,:]
            output_y0 = output_y0.reshape(batch_size1*input_channel)
            output_y0 = encode(output_y0)

            output_y1 = output[1::3,:]
            output_y1 = output_y1.reshape(batch_size1*input_channel)
            output_y1 = encode(output_y1)

            output_t = output[2::3,:]
            output_t = output_t.reshape(batch_size1*input_channel)
            output_t = encode(output_t)

            cipher_array = cipher*(batch_size1*input_channel)
            output_c = cipher_array()
            output_c = package(output_y0,output_y1,output_t,output_c)


            c_path='/home/andy/下载/libset10.so'
            so_lib=cdll.LoadLibrary(c_path)
            res_array = ctypes.c_bool*len(output_c)
            res = res_array()
            # # res = so_lib.verifyApi(output_cc[1],t[1],2,co[1],2)
            # if input_channel==1:
            #     so_lib.verifyApiBatch(res,output_c,len(output_c),t,co)
            # else:
            #     so_lib.verifyApiBatchFC(res,output_c,len(output_c),t,32,co,32)
            so_lib.verifyApiBatch(res, output_c, len(output_c),t,co)
            res_n = np.zeros([batch_size1*input_channel], dtype='O')
            res_n[:]=res
            res_n=res_n.reshape(batch_size1,input_channel)
            verify_res[:,:,w]=res_n

        return (outputs,verify_res)


class Fc_numpy(object):

    def __init__(self, in_channel, out_channel):
        # self.weight = np.float64(np.ones((in_channel, out_channel)) * 0.1)
        # self.weight = np.float64(np.random.randn(in_channel, out_channel) * 0.1)
        self.weight = np.zeros([out_channel,in_channel],dtype='O')
        self.bias = np.zeros((out_channel,), dtype='O')
        self.out_channel = out_channel
        self.weight_ = np.zeros([out_channel,in_channel],dtype='O')
        self.flag = False

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self,inputs):
        batch_size, in_channel = inputs.shape
        outputs = np.zeros([batch_size, self.out_channel], dtype='O')
        if self.bias is not None:
            for i in range(inputs.shape[0]):
                outputs[i] = np.dot(self.weight,inputs[i]) + self.bias
        else:
            if self.flag==False:
                for i in range(inputs.shape[0]):
                    outputs[i] = np.dot(self.weight,inputs[i])
            else:
                for i in range(inputs.shape[0]):
                    outputs[i] = np.dot(self.weight_,inputs[i])
        #verify
        outputs1 = outputs.copy()
        inputs = np.repeat(inputs[:, np.newaxis, :], self.out_channel, axis=1)
        weight_ = np.repeat(self.weight[np.newaxis, :, :], batch_size, axis=0)
        weight_ = weight_[2::3,:,:]
        batch_size1 = int(batch_size / 3)
        weight_ = weight_.reshape(batch_size1*self.out_channel,-1)

        c_array2 = ctypes.c_char_p * (in_channel)
        c_array1 = c_array2 * (self.out_channel*batch_size1)
        c_array3 = ctypes.c_long * (in_channel)
        c_array0 = c_array3 * (self.out_channel*batch_size1)

        input_t = inputs[2::3,:,:]
        input_t = input_t.reshape(batch_size1*self.out_channel,-1)
        encode = np.vectorize(int_charp)
        input_t = encode(input_t)
        t = c_array1()
        for i in range(len(t)):
            temp = c_array2()
            temp[:] = input_t[i]
            t[i] = temp

        co = c_array0()
        for i in range(len(co)):
            temp = c_array3()
            temp[:] = weight_[i]
            co[i] = temp

        output_y0 = outputs1[0::3,:]
        output_y0 = output_y0.reshape(batch_size1 * self.out_channel)
        output_y0 = encode(output_y0)

        output_y1 = outputs1[1::3, :]
        output_y1 = output_y1.reshape(batch_size1 * self.out_channel)
        output_y1 = encode(output_y1)

        output_t = outputs1[2::3, :]
        output_t = output_t.reshape(batch_size1 * self.out_channel)
        output_t = encode(output_t)

        cipher_array = cipher * (batch_size1 * self.out_channel)
        output_c = cipher_array()
        output_c = package(output_y0, output_y1, output_t, output_c)

        c_path = '/home/andy/下载/libset16.so'
        so_lib = cdll.LoadLibrary(c_path)
        res_array = ctypes.c_bool * len(output_c)
        res = res_array()
        res_n = np.zeros([batch_size1*self.out_channel], dtype='O')
        so_lib.verifyApiBatchFC(res,output_c,len(output_c),t,192,co,192)
        # else:
        #     so_lib.verifyApiBatchFC(res,output_c,len(output_c),t,512,co,512)
        # for i in range(len(output_c)):
        #     if in_channel==192:
        #         res[i]=so_lib.verifyApi(output_c[i],t[i],192,co[i],192)
        #     else:
        #         res[i]=so_lib.verifyApi(output_c[i],t[i],512,co[i],512)
        res_n[:] = res
        res_n=res_n.reshape(batch_size1,self.out_channel)
        # i=0
        # while i<len(output_c):
        #     if in_channel==192:
        #         block = 150
        #         if i+block<len(output_c):
        #             res_arrayb = ctypes.c_bool*(block)
        #             temp = res_arrayb()
        #             cipher_array_b = cipher*(block)
        #             output_c_b = cipher_array_b()
        #             output_c_b[:]=output_c[i:i+block]
        #             t_b_array = c_array2*block
        #             t_b=t_b_array()
        #             t_b[:]=t[i:i+block]
        #             co_b_array=c_array3*block
        #             co_b=co_b_array()
        #             co_b[:]=co[i:i+block]
        #             so_lib.verifyApiBatch(temp,output_c_b,block,t_b,192,co_b,192)
        #             res[i:i+block]=temp
        #         else:
        #             res_arrayb = ctypes.c_bool*(len(output_c)-i-1)
        #             temp = res_arrayb()
        #             cipher_array_b = cipher*(len(output_c)-i-1)
        #             output_c_b = cipher_array_b()
        #             output_c_b[:]=output_c[i:-1]
        #             t_b_array = c_array2*(len(output_c)-i-1)
        #             t_b=t_b_array()
        #             t_b[:]=t[i:-1]
        #             co_b_array=c_array3*(len(output_c)-i-1)
        #             co_b=co_b_array()
        #             co_b[:]=co[i:-1]
        #             so_lib.verifyApiBatch(temp,output_c_b,len(output_c)-i-1,t_b,192,co_b,192)
        #             res[i:-1]=temp
        #             res[-1]=so_lib.verifyApi(output_c[-1],t[-1],192,co[-1],192)
        #         i = i + block
        # so_lib.verifyApiBatch(res, output_c, len(output_c), t, 512,co,512)

        return (outputs,res_n)