import torch.nn as nn
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE","hospital_2.settings")
django.setup()
from hosptialapp.httptest import *
from hosptialapp.verify import *
import numpy as np



def int_charp(x):
    x=str(x)
    # x = x.encode()
    return x


def package(y0,y1,t):
    y0=y0.tolist()
    y1=y1.tolist()
    t=t.tolist()
    c = {'y0':y0,'y1':y1,'t':t}
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
        # self.kernel_8 = np.zeros((self.output_channel, input_channel, 2))
        self.kernel_size = kernel_size
        self.weight = np.random.randn(output_channel, input_channel, self.kernel_size)
        self.bias = True
        if bias:
            self.bias = np.random.randn(output_channel)

    def __call__(self, inputs, s):
        return self.infer(inputs, s)

    def infer(self, inputs, s):
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

        if self.dilation > 1:
            for i in range(self.kernel_size[0]):
                for j in range(self.kernel_size[1]):
                    kernel[:, :, self.dilation[0] * i, self.dilation[1] * j] = self.weight[:, :, i, j]
        else:
            kernel = self.weight

        batch_size1 = int(batch_size / 3)

        #verify
        if inputs_padding.shape[-1]%2==1:
            verify_in_all = inputs_padding[:,:,:-1:]
        verify_in_all1 = np.swapaxes(verify_in_all, 1, 2)
        verify_in_all = verify_in_all1[2::3,:,:]
        verify_in_1 = verify_in_all[:, 0::2, :]
        verify_in_2 = verify_in_all[:, 1::2, :]
        verify_in = np.concatenate((verify_in_1, verify_in_2), axis=2)
        to_char = np.vectorize(int_charp)
        verify_in = to_char(verify_in)
        if input_channel==1:
            print('conv1off start:')
            signal_on = send({'function':'verifyConvOFFBatch','data':verify_in.tolist(),'state':1},s)
            print('conv1off response:',signal_on['result'])
        else:
            print('conv2off start:')
            signal_on = send({'function':'verifyConvOFFBatch','data':verify_in.tolist(),'state':2},s)
            print('conv2off response:', signal_on['result'])



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

            # output shape: batch_size, output_channel
            output = input_ * kernel_
            output = np.sum(output, axis=(-1, -2))
            outputs[:,:,w] = output

        #verify
        # coefficient = np.swapaxes(kernel_, 1, 2)
        # coefficient_1 = coefficient[:, 0::2, :]
        # coefficient_2 = coefficient[:, 1::2, :]
        # coefficient = np.concatenate((coefficient_1, coefficient_2), axis=2).reshape(self.output_channel, -1)
        # if input_channel==1:
        #     np.savetxt('/home/www/project/hospital_2/hosptialapp/conv1', coefficient, fmt='%d', delimiter=',',newline='\n')
        # else:
        #     np.savetxt('/home/www/project/hospital_2/hosptialapp/conv2', coefficient, fmt='%d', delimiter=',',newline='\n')

        # post_data(verify_in)
        verify_out = np.swapaxes(outputs,1,2)
        verify_out_t = verify_out[2::3,:,:]
        verify_out_cipher = np.zeros([batch_size1,output_w,self.output_channel], dtype='O')
        for i in range(0,verify_out.shape[0],3):
            for j in range(verify_out.shape[1]):
                for k in range(verify_out.shape[2]):
                    verify_out_cipher[int(i/3)][j][k] = [str(verify_out[i][j][k]),str(verify_out[i+1][j][k]),str(verify_out[i+2][j][k])]
        # print(verify_out_cipher.shape)
        # print(verify_out_cipher)
        while(1):
            if(signal_on['result']=='copy'):
                res_json = send({'function':'verifyConvONBatch','cipher':verify_out_cipher.tolist(),'state':0},s)['result']
                print('conv_verify_result received:',res_json!=None)
                break
            else:
                continue

        # print('out\n')
        # print(verify_out)
        # print('in\n')
        # print(verify_in)
        # print('co\n')
        # print(coefficient)

        # for i in range(verify_out_t.shape[0]):
        #     for j in range(verify_out_t.shape[1]):
        #         for k in range(verify_out_t.shape[2]):
        #             print(verify_out_t[i][j][k]==np.sum(verify_in[i][j]*coefficient[k]))
        #             pass



        if self.bias is not None:
            bias_ = np.tile(self.bias.reshape(-1, 1), (1, output_w)). \
                reshape(self.output_channel, output_w)
            outputs += bias_
        return (outputs,res_json,1)

class Avgpool_numpy:
    def __init__(self,kernel_size,stride,padding=0,ceil_mode=False, count_include_pad=True):
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.ceil_mode=ceil_mode
        self.count_include_pad=count_include_pad

    def __call__(self, inputs, s):
        return self.infer(inputs,s)

    def infer(self,inputs,s):
        # 根据参数，算出输出的shape
        batch_size, input_channel, width = inputs.shape
        output_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        outputs = np.zeros([batch_size, input_channel, output_w], dtype='O')

        # 计算padding之后的inputs_array
        inputs_padding = np.zeros([batch_size, input_channel,  width + 2 * self.padding],dtype='O')
        inputs_padding[:, :, self.padding:self.padding + width] = inputs    #两边填充padding，将输入放在中间
        batch_size1 = int(batch_size / 3)
        #verify
        verify_in = inputs_padding[2::3,:,:]
        to_char = np.vectorize(int_charp)
        verify_in = to_char(verify_in)
        print('pool Off send')
        signal_on = send({'function':'verifyPoolOFFBatch','data':verify_in.tolist(),'state':3},s)
        print('pool Off response:',signal_on['result'])

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
        verify_out = outputs
        verify_out_cipher = np.zeros([batch_size1, input_channel, output_w], dtype='O')
        for i in range(0, verify_out.shape[0], 3):
            for j in range(verify_out.shape[1]):
                for k in range(verify_out.shape[2]):
                    verify_out_cipher[int(i / 3)][j][k] = [str(verify_out[i][j][k]), str(verify_out[i + 1][j][k]),
                                                           str(verify_out[i + 2][j][k])]
        # print(verify_out_cipher.shape)

        while(1):
            if(signal_on['result']=='copy'):
                res_json = send({'function':'verifyPoolONBatch','cipher':verify_out_cipher.tolist(),'state':0},s)['result']
                print('pool on response received:',res_json!=None)
                break
            else:
                continue

        return (outputs,res_json,1)


class Fc_numpy(object):

    def __init__(self, in_channel, out_channel):
        # self.weight = np.float64(np.ones((in_channel, out_channel)) * 0.1)
        # self.weight = np.float64(np.random.randn(in_channel, out_channel) * 0.1)
        self.weight = np.zeros([out_channel,in_channel],dtype='O')
        self.bias = np.zeros((out_channel,), dtype='O')
        self.out_channel = out_channel

    def __call__(self, inputs, s):
        return self.forward(inputs, s)

    def forward(self,inputs,s):
        batch_size, in_channel = inputs.shape

        #verify
        # for i in range(0,inputs.size(),200):
        to_char = np.vectorize(int_charp)
        verify_in = to_char(inputs)
        verify_in = verify_in[2::3,:]
        print('FC Off start')
        signal_on = send({'function':'verifyFCOFFBatch','data':verify_in.tolist(),'state':4},s)
        print('FC Off response:',signal_on['result'])
        outputs = np.zeros([batch_size, self.out_channel], dtype='O')
        if self.bias is not None:
            for i in range(inputs.shape[0]):
                outputs[i] = np.dot(self.weight,inputs[i]) + self.bias
        else:
            for i in range(inputs.shape[0]):
                outputs[i] = np.dot(self.weight,inputs[i])
        verify_out_cipher = np.zeros([int(batch_size/3), self.out_channel], dtype='O')
        for i in range(0, outputs.shape[0], 3):
            for j in range(outputs.shape[1]):
                verify_out_cipher[int(i / 3)][j] = [str(outputs[i][j]), str(outputs[i + 1][j]),
                                                           str(outputs[i + 2][j])]
        # print(verify_out_cipher.shape)

        while(1):
            if(signal_on['result']=='copy'):
                res_json = send({'function':'verifyFCONBatch','cipher':verify_out_cipher.tolist(),'state':4},s)['result']
                print('FC On response received:',res_json!=None)
                break
            else:
                continue

        return (outputs,res_json, 1)