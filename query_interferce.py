from hosptialapp.range import example_query
from hosptialapp.range import interface
from hosptialapp.keyword import keyword_interface_impl
import pandas as pd
from datetime import datetime
import numpy as np
import re

def cut(patient):
    totalCount = re.sub("\D", "", patient)
    return totalCount
def range_query(dimension,lower_bound,upper_bound):
    result_path = interface.range_query(dimension,lower_bound,upper_bound,'/home/www/project/hospital_2/hosptialapp/range/key','/home/www/project/hospital_2/hosptialapp/range/output')
    coloum=list(range(4))
    query_result = pd.read_csv(result_path,index_col=False,names=coloum,header=None)
    msg = []
    # print('values',query_result.values)
    for i in query_result.values:
        msg.append({'id':i[0],'age':i[1],'time':datetime.fromtimestamp(i[2]).strftime('%Y-%m-%d %H:%M:%S'),'result':int(i[3])})
    return msg

def range_query_verify():
    workPath = "/home/www/project/hospital_2/hosptialapp/range/output"
    verify_path = interface.range_verify(workPath)
    coloum=[1]
    query_result = pd.read_csv(verify_path,index_col=False,names=coloum,header=None,dtype=str)
    res = 0
    error_message = []
    print('values:',type(query_result.values),query_result.values.shape)
    if len(query_result.values)>1:
        for i in range(1,len(query_result.values)):
            error_message.append(query_result.values[i][0])
    if query_result.values[0][0] == 'UNPASS':
        pass
    else:
        res=1
    return (res,error_message)

def keyword_query(query_keywords):
    # query_keywords = ["鼻塞", "填充剂", "131碘-MIBG恶性肿瘤治疗"]
    result_csv_file_path = keyword_interface_impl.keyword_query(key_words=query_keywords, keyFilePath="/home/www/project/hospital_2/hosptialapp/keyword/sk.bin", workPath="/home/www/project/hospital_2/hosptialapp/keyword")
    # 上述查询过程可能要持续1分钟不等（看网络情况），只要服务器端界面在弹出日志，就是运行正常
    result = np.loadtxt(result_csv_file_path,dtype='str')

    print('result:',result)
    print('type:',type(result))
    print('shape',result.shape)
    # if len(result)==0:
    #     msg = []
    #     verify_result = {'result':'unpassed','message':'没有查询到，输入样例错误'}
    #     return (msg,verify_result)
    cuter = np.vectorize(cut)
    result = cuter(result)
    msg = []
    result = result.tolist()
    print('1',result)
    print('2',type(result))
    if len(query_keywords)==4 and query_keywords[-1]!='':
            msg.append({'id':result})
    else:
        for i in result:
            msg.append({'id':i})

    print("查询结果文件路径：", result_csv_file_path)
    verify_result = keyword_interface_impl.keyword_verify(key_words=query_keywords, keyFilePath="/home/www/project/hospital_2/hosptialapp/keyword/sk.bin", workPath="/home/www/project/hospital_2/hosptialapp/keyword",
                                   resultCSVPath=result_csv_file_path)
    # verify_result={'result':'通过','message':'pass'}
    print(verify_result)
    print('key',query_keywords)
    print('len',len(query_keywords))
    print(result)
    return (msg,verify_result)

def range_alter(id,age,time_str,label):
    id = int(id)
    print('id',id)
    coloum=list(range(4))
    result_path = "/home/www/project/hospital_2/hosptialapp/range/output/result.csv"
    query_result = pd.read_csv(result_path,index_col=0,names=coloum,header=None)
    if age!=None:
        query_result.loc[id,1] = float(age)
    if time_str!=None:
        query_result.loc[id,2] = datetime.strptime(time_str,'%Y-%m-%d %H:%M:%S').timestamp()
    if label!=None:
        query_result.loc[id,3] = float(label)
    # if id_2!=None:
    #     query_result.drop(id_1,inplace=True)
    query_result.to_csv(result_path,header=False)

def delete_range(id):
    coloum=list(range(4))
    result_path = "/home/www/project/hospital_2/hosptialapp/range/output/result.csv"
    print('删除的id',id)
    query_result = pd.read_csv(result_path,index_col=0,names=coloum,header=None)
    query_result.drop(labels=int(id), inplace=True,axis=0)
    query_result.to_csv(result_path,header=False)

def key_alter(id_b,id_a):
    patients = []
    with open('/home/www/project/hospital_2/hosptialapp/keyword/keyword.result.txt','r') as f:
        data = f.readlines()
        print('改之前：',data)
        for i in range(len(data)):
            if id_b==data[i][2:-1]:
                if i!=len(data)-1:
                    temp = '患者'+id_a+'\n'
                else:
                    temp = '患者'+id_a
                patients.append(temp)
                print('改的：',data[i])
            else:
                patients.append(data[i])
        print('改之后：',patients)
        f.close()
    with open('/home/www/project/hospital_2/hosptialapp/keyword/keyword.result.txt','w') as f:
        f.writelines(patients)

def key_delete(id):
    patients = []
    with open('/home/www/project/hospital_2/hosptialapp/keyword/keyword.result.txt','r') as f:
        data = f.readlines()
        print('删之前：',data)
        for i in range(len(data)):
            print('id:',id)
            print('data_i',data[i][2:-1])
            if id==data[i][2:-1]:
                if i!=len(data)-1:
                    continue
                else:
                    if i==0:
                        continue
                    else:
                        patients[i-1] = patients[i-1][:-1]
                print('删的：',data[i])
            else:
                patients.append(data[i])
        print('删之后：',patients)
        f.close()
    with open('/home/www/project/hospital_2/hosptialapp/keyword/keyword.result.txt','w') as f:
        f.writelines(patients)

def key_verify(keywords):
    verify_result = keyword_interface_impl.keyword_verify(key_words=keywords, keyFilePath="/home/www/project/hospital_2/hosptialapp/keyword/sk.bin", workPath="/home/www/project/hospital_2/hosptialapp/keyword",
                                   resultCSVPath='/home/www/project/hospital_2/hosptialapp/keyword/keyword.result.txt')
    return verify_result




