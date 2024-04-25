import csv
import datetime

import pandas as pd
import torch
from hosptialapp.verifiable_cnn1 import *
os.environ.setdefault("DJANGO_SETTINGS_MODULE","hospital_2.settings")
django.setup()
from hosptialapp import models
import time
import requests


def sdigit(x):
    return round(x)

def verify_handle(verify_res,start,layer_num):
    locations = []
    for key,value in verify_res.items():
        if value=='0':
            locations.append(key)
    if len(locations)==0:
        return 0
    else:
        for location in locations:
            location = location.split(':')
            id = start+int(location[0])

def result_handle(x,start):
    msg = {}
    for i in range(0,x.shape[0],3):
        temp = []
        temp.append(x[i].tolist())
        temp.append(x[i+1].tolist())
        temp.append(x[i+2].tolist())
        msg.update({'l' + str(int(i / 3)+start): temp})
    return msg

def data_insert(verify,layer,data_all,equation_start_id,user_start_id):
    equation_type = [0]
    if layer==1 or layer==2 or layer==4:
        equation_type[0] = 2
    elif layer==3:
        equation_type[0] = 32
    else:
        equation_type[0] = 192
        i = 0
        for key,value in verify.items():
            eid = equation_start_id+i
            uid = user_start_id+int(key[0])
            i = i+1
            data_all.loc[len(data_all.index)] = [eid,equation_type[0],layer,value,uid]
            pass

def interfer_all():
    url = "http://127.0.0.1:1316"
    session = requests.session()
    torch.set_printoptions(precision=6)

    pd.set_option('max_colwidth',200)
    cnn=CNN()
    cnn.load_state_dict(torch.load('/home/www/project/hospital_2/hosptialapp/model_parameter.pkl3'))

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
            elif alter_data.layer_no == '2':
                cnn_numpy2.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 0] = int(alter_data.num1)
                cnn_numpy2.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 1] = int(alter_data.num2)
            elif alter_data.layer_no == '3':
                fc_2_numpy.weight[int(alter_data.output_no), int(alter_data.input_no)] = int(alter_data.weight)
    except:
        pass


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_dir = '/home/data/'
    path_enc = './encrypteds.csv'
    coloum=list(range(617))
    log_dir = '/home/www/project/hospital_2/hosptialapp/log/'

    start = 1
    end = 100000
    #batch_count = int((end-start+1)%1000/100)
    # file_count = int((end-start+1)/1000)

    # if (end-start+1)%1000>0:
    #     file_count = file_count+1
    # if (end-start+1)%1000%100>0:
    #     batch_count = batch_count+1
    batch_count = 100
    file_count = 300
    data_index = 1
    time1 = time.time()
    time_all = 0
    pd_columns = ['equation_id','equation_type','layer_id','verify_res','user_id']

    for file_index in range(1,file_count+1):
        file_path = path_dir+'data'+str(file_index)+'.csv'
        csv_path = log_dir+'data'+str(file_index)+'.csv'
        batch = 0
        equation_number = 0
        chunk_all = pd.read_csv(file_path, index_col=0,names=coloum,header=None,chunksize=100)
        pd_log = pd.DataFrame(None,columns=pd_columns)
        pd_log.set_index(['equation_id'], inplace=True)
        pd_log.to_csv(csv_path)

        for data in chunk_all:
            # if batch>0:
            #     break
            if file_index == file_count and batch==batch_count-1:
                break

            print('第%d个文件第%d个batch开始'%(file_index,batch))
            all_data=[]
            for i in data.values:
                y0=list(map(int,i[0:205]))
                y1=list(map(int,i[205:410]))
                t=list(map(int,i[410:615]))
                signal_enc=[y0,y1,t]    #一个二维列表，其中每个元素是每条数据的y0部分（一个列表）和每条数据的y1部分（一个列表）
                all_data+=signal_enc   #一个二维列表，前二个元素是第一条数据的y0部分和y1部分（每个都是列表），第3，4个元素是第二条数据的y0,y1部分，依次类推

            all_data=np.array(all_data)

            all_data=all_data.reshape([-1,1,205])

            time3 = time.time()
            s = 0
            x,verify_conv1,s=cnn_numpy(all_data,session)
            while s == 0:
                continue

            # with open(csv_path, 'a') as file:
            #     writer = csv.writer(file, lineterminator='\n')
            #     for key, value in verify_conv1.items():
            #             uid = batch*1000 + int(key[0])
            #             # pd_log_temp = pd.DataFrame([[equation_number, 2, 1, value, uid]])
            #             # pd_log_temp.to_csv(csv_path, mode='a')
            #             writer.writerow([equation_number, 2, 1, value, uid])
            #             equation_number = equation_number + 1
            #             # pd_log.loc[len(pd_log.index)] = [eid, 2, 1, value, uid]
            #     file.flush()
            #     file.close()
            # pass
            # data_insert(verify_conv1,1,pd_log,data_index,batch*100)

            s = 0

            x,verify_pool1,s=pool_numpy(x,session)
            # with open(csv_path, 'a') as file:
            #     writer = csv.writer(file, lineterminator='\n')
            #     for key, value in verify_pool1.items():
            #         # with open(csv_path,'a') as file:
            #         #     writer1 = csv.writer(file,lineterminator='\n')
            #             uid = batch*1000 + int(key[0])
            #             equation_number = equation_number + 1
            #             writer.writerow([equation_number, 2, 2, value, uid])
            #     file.flush()


            while s == 0:
                continue

            s=0
            x,verify_conv2,s=cnn_numpy2(x,session)

            # with open(csv_path, 'a') as file:
            #     writer = csv.writer(file, lineterminator='\n')
            #     for key, value in verify_conv2.items():
            #         # with open(csv_path,'a') as file:
            #         #     writer = csv.writer(file,lineterminator='\n')
            #             uid = batch*1000 + int(key[0])
            #             equation_number = equation_number + 1
            #             writer.writerow([equation_number, 32, 3, value, uid])
            #     file.flush()

            while s == 0:
                continue
            s=0
            x,verify_pool2,s=pool_numpy(x,session)
            # with open(csv_path, 'a') as file:
            #     writer = csv.writer(file, lineterminator='\n')
            #     for key, value in verify_pool2.items():
            #         # with open(csv_path,'a') as file:
            #         #     writer = csv.writer(file,lineterminator='\n')
            #             uid = batch*1000 + int(key[0])
            #             equation_number = equation_number + 1
            #             writer.writerow([equation_number, 2, 4, value, uid])
            #     file.flush()

            x=x.reshape([-1,192])

            while s == 0:
                continue

            s = 0

            x,verify_fc2,s=fc_2_numpy(x,session)

            while s == 0:
                continue
            # with open(csv_path, 'a') as file:
            #     writer = csv.writer(file, lineterminator='\n')
            #     for key, value in verify_fc2.items():
            #         # with open(csv_path,'a') as file:
            #         #     writer = csv.writer(file,lineterminator='\n')
            #             uid = batch*1000 + int(key[0])
            #             equation_number = equation_number + 1
            #             writer.writerow([equation_number, 192, 5, value, uid])
            #     file.flush()

                # handle_verify(start,end,verify_conv1,verify_pool1,verify_conv2,verify_pool2,verify_fc2)
            print('第%d个文件第%d个batch结束'%(file_index,batch))
            time4 = time.time()
            batch = batch+1
            data_index = data_index+1000
            time_all = time_all+(time4-time3)


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
    time2 = time.time()
    print('总时间：',time2-time1,'s')
    print('计算验证所花总时间：',time_all,'s')
    pass

if __name__=='__main__':
    url = "http://127.0.0.1:1316/post"
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
            elif alter_data.layer_no == '2':
                cnn_numpy2.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 0] = int(alter_data.num1)
                cnn_numpy2.weight[int(alter_data.kernel_no), int(alter_data.channel_no), 1] = int(alter_data.num2)
            elif alter_data.layer_no == '3':
                fc_2_numpy.weight[int(alter_data.output_no), int(alter_data.input_no)] = int(alter_data.weight)
    except:
        pass


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_dir = '/home/data/'
    path_enc = './encrypteds.csv'
    coloum=list(range(617))
    log_dir = '/home/www/project/hospital_2/hosptialapp/log/'

    start = 1
    end = 100000
    batch_count = int((end-start+1)%1000/100)
    file_count = int((end-start+1)/1000)
    # if (end-start+1)%1000>0:
    #     file_count = file_count+1
    # if (end-start+1)%1000%100>0:
    #     batch_count = batch_count+1
    data_index = 1
    time1 = time.time()
    time_all = 0
    pd_columns = ['equation_id','equation_type','layer_id','verify_res','user_id']

    file_count = 300
    batch_count = 1
    print(batch_count)
    print(file_count)
    for file_index in range(1,file_count+1):
        file_path = path_dir+'data'+str(file_index)+'.csv'
        csv_path = log_dir+'data'+str(file_index)+'.csv'
        batch = 0
        equation_number = 0
        chunk_all = pd.read_csv(file_path, index_col=0,names=coloum,header=None,chunksize=100)
        pd_log = pd.DataFrame(None,columns=pd_columns)
        pd_log.set_index(['equation_id'], inplace=True)
        pd_log.to_csv(csv_path)

        for data in chunk_all:
            if batch>1:
                break
            if file_index == file_count and batch==batch_count-1:
                break

            print('第%d个文件第%d个batch开始'%(file_index,batch))
            all_data=[]
            for i in data.values:
                y0=list(map(int,i[0:205]))
                y1=list(map(int,i[205:410]))
                t=list(map(int,i[410:615]))
                signal_enc=[y0,y1,t]    #一个二维列表，其中每个元素是每条数据的y0部分（一个列表）和每条数据的y1部分（一个列表）
                all_data+=signal_enc   #一个二维列表，前二个元素是第一条数据的y0部分和y1部分（每个都是列表），第3，4个元素是第二条数据的y0,y1部分，依次类推

            all_data=np.array(all_data)

            all_data=all_data.reshape([-1,1,205])

            time3 = time.time()
            s = 0
            x,verify_conv1,s=cnn_numpy(all_data,session)
            while s == 0:
                continue

            # with open(csv_path, 'a') as file:
            #     writer = csv.writer(file, lineterminator='\n')
            #     for key, value in verify_conv1.items():
            #             uid = batch*1000 + int(key[0])
            #             # pd_log_temp = pd.DataFrame([[equation_number, 2, 1, value, uid]])
            #             # pd_log_temp.to_csv(csv_path, mode='a')
            #             writer.writerow([equation_number, 2, 1, value, uid])
            #             equation_number = equation_number + 1
            #             # pd_log.loc[len(pd_log.index)] = [eid, 2, 1, value, uid]
            #     file.flush()
            #     file.close()
            # pass
            # data_insert(verify_conv1,1,pd_log,data_index,batch*100)

            s = 0

            x,verify_pool1,s=pool_numpy(x,session)
            # with open(csv_path, 'a') as file:
            #     writer = csv.writer(file, lineterminator='\n')
            #     for key, value in verify_pool1.items():
            #         # with open(csv_path,'a') as file:
            #         #     writer1 = csv.writer(file,lineterminator='\n')
            #             uid = batch*1000 + int(key[0])
            #             equation_number = equation_number + 1
            #             writer.writerow([equation_number, 2, 2, value, uid])
            #     file.flush()


            while s == 0:
                continue

            s=0
            x,verify_conv2,s=cnn_numpy2(x,session)

            # with open(csv_path, 'a') as file:
            #     writer = csv.writer(file, lineterminator='\n')
            #     for key, value in verify_conv2.items():
            #         # with open(csv_path,'a') as file:
            #         #     writer = csv.writer(file,lineterminator='\n')
            #             uid = batch*1000 + int(key[0])
            #             equation_number = equation_number + 1
            #             writer.writerow([equation_number, 32, 3, value, uid])
            #     file.flush()

            while s == 0:
                continue
            s=0
            x,verify_pool2,s=pool_numpy(x,session)
            # with open(csv_path, 'a') as file:
            #     writer = csv.writer(file, lineterminator='\n')
            #     for key, value in verify_pool2.items():
            #         # with open(csv_path,'a') as file:
            #         #     writer = csv.writer(file,lineterminator='\n')
            #             uid = batch*1000 + int(key[0])
            #             equation_number = equation_number + 1
            #             writer.writerow([equation_number, 2, 4, value, uid])
            #     file.flush()

            x=x.reshape([-1,192])

            while s == 0:
                continue

            s = 0

            x,verify_fc2,s=fc_2_numpy(x,session)

            while s == 0:
                continue
            # with open(csv_path, 'a') as file:
            #     writer = csv.writer(file, lineterminator='\n')
            #     for key, value in verify_fc2.items():
            #         # with open(csv_path,'a') as file:
            #         #     writer = csv.writer(file,lineterminator='\n')
            #             uid = batch*1000 + int(key[0])
            #             equation_number = equation_number + 1
            #             writer.writerow([equation_number, 192, 5, value, uid])
            #     file.flush()

                # handle_verify(start,end,verify_conv1,verify_pool1,verify_conv2,verify_pool2,verify_fc2)
            print('第%d个文件第%d个batch结束'%(file_index,batch))
            time4 = time.time()
            batch = batch+1
            data_index = data_index+1000
            time_all = time_all+(time4-time3)


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
    time2 = time.time()
    print('总时间：',time2-time1,'s')
    print('计算验证所花总时间：',time_all,'s')
    pass