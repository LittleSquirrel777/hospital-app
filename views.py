import os,django
import subprocess

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "../hospital_2.settings")
django.setup()
import datetime
import json
import math
import os
import pickle
import webbrowser
from time import sleep
import random
import pytz
from django.http import JsonResponse, FileResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from reportlab.platypus import SimpleDocTemplate, Spacer
from rest_framework import serializers
#from rest_framework_jwt.serializers import jwt_encode_handler,jwt_payload_handler
from UTC import TPAServer
from hosptialapp import models
# Create your views here.
from django.middleware.csrf import get_token
from django.utils import formats,timezone
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, Circle
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Spacer, SimpleDocTemplate, Table, Paragraph, Image
from reportlab.graphics.shapes import Image as DrawingImage
from hosptialapp.pdf import myFirstPage,myLaterPages
from hosptialapp.forward_1 import *
from hosptialapp.plain_interfer import *
from UTC import checkError
from hosptialapp.query_interferce import *
from hosptialapp.batch_forward import *
from datetime import datetime
# 全局变量
num=0

def getcsrf(request):
    return JsonResponse({'csrftoken': get_token(request) or 'NOTPROVIDED'})


def chooselogin(request):
    return render(request,"index.html")
def docterlogin(request):
    return render(request,"doctorlogin.html")
def adminlogin(request):
    return render(request,"adminlogin.html")
# def register(request):
#     user=request.POST.get("user")
#     password=request.POST.get('password')
#     id=models.Userinfo.objects.latest("id").id+1
#     userinfo=models.Userinfo(id=id,username=user,password=password)
#     userinfo.save()
#     return JsonResponse({'codestatus':1},safe=False)
@csrf_exempt
def login(request):
    username=request.POST.get('username')
    password=request.POST.get('password')
    identity = request.POST.get('identity')
    if username and password:
        try:
            user=models.Userinfo.objects.get(username=username)
        except:
            return JsonResponse({'code':500,'message': '登陆失败'},safe=False)
        if user.password==password:
            if user.identity=='3':
                token='admin-token'
            else:
                token='editor-token'
            # payload = jwt_payload_handler(user)
            # token = jwt_encode_handler(payload)
            return JsonResponse({'code':200, 'token':token,'message': '登陆成功'},safe=False)
        else:
            return JsonResponse({'code': 400, 'message': '用户名或密码错误'}, safe=False)
    else:
        return JsonResponse({'code': 400, 'message': '请填写用户名和密码'}, safe=False)


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Userinfo
        fields = '__all__'
@csrf_exempt
def info(request):
    username=request.GET.get('username')
    if username:
        try:
            user=models.Userinfo.objects.get(username=username)
        except:
            return JsonResponse({'code':500},safe=False)
        serializer =UserProfileSerializer(user)
        return JsonResponse({'code':200,'user':serializer.data,'identity':user.identity},safe=False)
        # return JsonResponse({'roles': roles}, safe=False)
    return JsonResponse({'code':404},safe=False)
def getrole(request):
    return JsonResponse({'code':200,'roles':['admin','editor']},safe=False)
@csrf_exempt
def logout(request):
    return JsonResponse({'code': 200}, safe=False)
def addrecord(request):
    name = request.POST.get('patient_name')
    gender = request.POST.get('patient_gender')
    age = request.POST.get('patient_age')
    desc = request.POST.get('disease_description')
    # time = request.POST.get('disease_time')
    otherdata = request.POST.get('otherdata')
    id = models.medical_record.objects.latest("id").id + 1
    # print(age)
    try:
        record = models.medical_record(id=id, name=name, gneder=gender, age=age, desc=desc,
                                       otherdata=otherdata)
        record.save()
        return JsonResponse({'status': 1}, safe=False)
    except:
        return JsonResponse({'status':2},safe=False)
@csrf_exempt
def addrecord1(request):
    name = request.POST.get('name')
    gender = request.POST.get('gender')
    age = request.POST.get('age')
    remark = request.POST.get('remark')
    result = request.POST.get('result')
    time = request.POST.get('time')
    feature1 = request.POST.get('feature1')
    feature2 = request.POST.get('feature2')
    status =request.POST.get('status')
    id = models.medicalrecord.objects.latest("id").id + 1
    if time!=None:
       t=datetime.strptime(time,"%Y-%m-%dT%H:%M:%S.%fZ")
    else:
       t=datetime.now()
        # t=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        record = models.medicalrecord(id=id,name=name, gender=gender, age=age, result=result,remark=remark,
                                       time=t,status=status,feature1=feature1,feature2=feature2)
        record.save()
        return JsonResponse({'code': 200,'message':'病历创建成功'}, safe=False)
    except:
        return JsonResponse({'code':500,'message':'病历创建失败'},safe=False)
class MedicalRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.medicalrecord
        fields = '__all__'
def get_all_record(request):
    list=models.medicalrecord.objects.all().order_by("-id")
    total=len(list)
    serializer = MedicalRecordSerializer(list, many=True)
    # print(serializer.data)
    return JsonResponse({'code': 200, 'items': serializer.data, 'total': total}, safe=False)
def get_record_id(request):
    id=request.GET.get('id')
    res=models.medicalrecord.objects.get(id=id)
    if res==None:
        return JsonResponse({'code': 500}, safe=False)
    serializer = MedicalRecordSerializer(res)
    return JsonResponse({'code': 200, 'data': serializer.data}, safe=False)
@csrf_exempt
def update_record(request):
    name = request.POST.get('name')
    gender = request.POST.get('gender')
    age = request.POST.get('age')
    remark = request.POST.get('remark')
    result = request.POST.get('result')
    time = request.POST.get('time')
    feature1 = request.POST.get('feature1')
    feature2 = request.POST.get('feature2')
    status = request.POST.get('status')
    id = request.POST.get('id')
    # if feature2==" " and feature1==" " and
    #res为受影响的行数
    res=models.medicalrecord.objects.filter(id=id).update(name=name, gender=gender, age=age, remark=remark,
                                       result=result,time=time,feature2=feature2,feature1=feature1,status=status)
    if res==0:
        return JsonResponse({'code': 400},safe=False)
    else:
        return JsonResponse({'code': 200},safe=False)
def verify_id(request):
    # TODO:verify alg
    id = request.GET.get('id')
    global num
    print(num)
    # msg=infer_interface(id,id)
    # msg.update((('code',200),('error_node',2),('num',num)))
    msg={'code':200,'error_node':2,'num':num}
    # while num<=100 :
    #     num+=1
    #     sleep(0.1)
    res = models.medicalrecord.objects.get(id=id)
    res.status='verified'
    return JsonResponse(msg,safe=False)
@csrf_exempt
def interfer_batch(request):
    # TODO:verify alg
    id_list=[]
    print('verify')
    request_list=request.POST
    print(request_list)
    for i in range(0,len(request_list)):
        id_list.append(int(request_list.get('ids['+str(i)+']')))
    # print(id_list)
    from_=min(id_list)
    to_=max(id_list)
    print('收到客户端计算推理请求！')
    print('推理起始id:%d  推理截止id:%d'%(from_,to_))
    print('正在推理中......')

    # ids=request.body.getlist('ids')
    msg=infer_interface(from_,to_)
    print('推理完成！')
    print('interfer idlist',id_list)
    # msg={'code':200,'error_node':2,'l16':[[1,2,3,4],[6,7,8,9],[10,11,12,13]]}
    msg.update({'code':200})
    # if len(id_list)==1:
    #     try:
    #         record=models.medicalrecord.objects.get(id=id_list[0])
    #         verify_res=record.verify_res
    #         verify_error_layer=record.verify_error_layer
    #         print('---------------------------')
    #         print(verify_res)
    #         print(verify_error_layer)
    #     except:
    #         msg.update(('code',500))
    # print(ids)
    # 验证完成将状态修改为verified!
    # for id in id_list:
    # res = models.medicalrecord.objects.get(id=id_list[0])
    # res.status='verified'
    return JsonResponse(msg,safe=False)

@csrf_exempt
def verify_batch(request):
    # TODO:verify alg
    id_list=[]
    print('verify')
    request_list=request.POST
    print(request_list)
    for i in range(0,len(request_list)):
        id_list.append(int(request_list.get('ids['+str(i)+']')))
    print(id_list)
    from_=min(id_list)
    to_=max(id_list)
    print(from_,to_)
    medicalrecord = models.medicalrecord
    layer_no = {}
    kernel_no = {}
    for i in id_list:
        layer_i = medicalrecord.objects.get(id=i).verify_error_layer
        kernel_i = medicalrecord.objects.get(id=i).verify_error_kernel
        layer_no.update({i:layer_i})
        kernel_no.update({i:kernel_i})
    print(layer_no)
    print(kernel_no)
    return JsonResponse({'code':200,'msg':'验证完成','layer_no':layer_no,'kernel_no':kernel_no})


@csrf_exempt
def result_write(request):
    id_list=[]
    res_list=[]
    request_list=request.POST
    # print('result')
    # print(request_list)
    for i in range(0,int(len(request_list)/2)):
        id_list.append(int(request_list.get('ids['+str(i)+']')))
        res_list.append(int(request_list.get('res['+str(i)+']')))
    # print(id_list)
    # print(res_list)
    msg={'code':200}
    dis_list=['NO1','NO2','NO3','NO4']
    try:
        for i in range(0,len(id_list)):
            result=str(dis_list[res_list[i]])
            res = models.medicalrecord.objects.filter(id=id_list[i]).update(status='verified',result=result)
    except:
        msg={'code':500}
    return JsonResponse(msg,safe=False)
#原本的存储验证方案
@csrf_exempt
def storage_verification1(request):
    # TODO:verify alg
    node_num = request.POST.get('node_num')
    fid=request.POST.get('fid')
    uid=request.POST.get('uid')

    print(node_num)
    TPAServer.challenge(fid,uid,metdNum=0,CNum=node_num)
    with open(r"//home/www/project/hospital_2/UTC/result.txt", "r", encoding='utf-8') as f:
        bool = f.readline()
    if bool == "True":
        bool = False
    else:
        bool = True
    print(bool)

    ##—————————————————————————在下添加代码———————————————————————
    ##如果bool == True（损坏）
    ##调用checkError，默认为22文件
    ##从checkResult.txt中取损坏的号
    ##—————————————————————————在上添加代码———————————————————————

    with open('/home/www/project/hospital_2/UTC/selected_list.txt', 'r') as file:
        # 逐行读取文件内容，并将每行内容转换为整数，存储在列表中
        my_list = [int(line.strip()) for line in file]
    print(my_list)
    msg={'code':200,'has_error':bool,'list':my_list}
    # msg={'code':200,'has_error':bool}
    return JsonResponse(msg,safe=False)
#polar存储验证方案
@csrf_exempt
def storage_verification(request):
    # TODO:verify alg
    node_num = int(request.POST.get('node_num'))
    # fid=request.POST.get('fid')
    # uid=request.POST.get('uid')
    random_number =random.sample(range(0,908),node_num)
    with open('/root/raymond/courage/la-por/build/indexFile','w') as file:
        for number in random_number:
            file.write(str(number)+ '\n')
    os.system('/root/raymond/courage/la-por/build/bin/clientBatch /root/raymond/courage/la-por/build/indexFile /home/data500G/ClientConfig/ /home/data500G/MerkleConfig/ -a')
    ##—————————————————————————在下添加代码———————————————————————
    ##如果bool == True（损坏）
    ##调用checkError，默认为22文件
    ##从checkResult.txt中取损坏的号
    ##—————————————————————————在上添加代码———————————————————————

    with open('/root/raymond/courage/la-por/build/resultFile.txt', 'r') as file:
        # 逐行读取文件内容，并将每行内容转换为整数，存储在列表中
        words = file.read()
    result=words.split()[-1]
    if result=='PASSED':
        bool=False
    else:
        bool=True
    print(result)
    # msg={'code':200,'has_error':bool,'list':my_list}
    # msg={'code':200,'has_error':bool}
    return JsonResponse({'code':200,'has_error':bool,'list':random_number},safe=False)
@csrf_exempt
def register(request):
    user=request.POST.get("username")
    password=request.POST.get('password')
    identity=request.POST.get('identity')
    email=request.POST.get('email')
    id=models.Userinfo.objects.latest("id").id+1
    if user and password and identity:
        try:
            res = models.Userinfo.objects.filter(username=user)
            if res.exists():
                return JsonResponse({'code': 400, 'message': '用户已存在'})
            else:
               userinfo=models.Userinfo(id=id,username=user,password=password,identity=identity,email=email)
               userinfo.save()
               return JsonResponse({'code':200,'message':'注册成功'},safe=False)
        except:
            return JsonResponse({'code':400,'message':'注册失败'},safe=False)
    else:
        return JsonResponse({'code': 400, 'message': '请填写用户名和密码'}, safe=False)
@csrf_exempt
def update(request):
    username=request.POST.get('username')
    email=request.POST.get('email')
    if username:
        res=models.Userinfo.objects.filter(username=username)
        if res.exists():
            user=models.Userinfo.objects.get(username=username)
            user.email=email
            user.save()
            return JsonResponse({'code':200,'message':'信息修改成功'},safe=False)
        else:
            return JsonResponse({'code':400,'message':'该用户不存在'},safe=False)
    else:
        return JsonResponse({'code': 400, 'message': '用户名不能为空'}, safe=False)
@csrf_exempt
def patient_list(request):
    result=request.GET.get('result')
    username=request.GET.get('username')
    sort=request.GET.get('sort')
    page=request.GET.get('page')
    limit=request.GET.get('limit')
    list=models.medicalrecord.objects.all().filter(name=username)
    if result:
        list=list.filter(result=result)
    if sort=='-id':
        list=list.order_by('-id')
    total = len(list)
    data=[]
    for item in list:
        dict={}
        dict['id']=item.id
        dict['name'] = item.name
        dict['result'] = item.result
        dict['remark'] = item.remark
        dict['age']=item.age
        dict['gender'] = item.gender
        dict['time'] = item.time
        dict['feature1'] = item.feature1
        dict['feature2'] = item.feature2
        dict['status']=item.status
        data.append(dict)
    return JsonResponse({'code': 200, 'items': data, 'total': total}, safe=False)
@csrf_exempt
def update_status(request):
    id=request.POST.get('id')
    status=request.POST.get('status')
    if id:
        record=models.medicalrecord.objects.get(id=id)
        record.status=status
        record.save()
        return JsonResponse({'code':200,'message':'状态修改成功'},safe=False)
    else:
        return JsonResponse({'code': 400, 'message': 'id不能为空'}, safe=False)
@csrf_exempt
def update_bypatient(request):
    name = request.POST.get('name')
    gender = request.POST.get('gender')
    age = request.POST.get('age')
    remark = request.POST.get('remark')
    result = request.POST.get('result')
    # time = request.POST.get('time')
    status = request.POST.get('status')
    id = request.POST.get('id')
    # if feature2==" " and feature1==" " and
    #res为受影响的行数
    res=models.medicalrecord.objects.filter(id=id).update(name=name, gender=gender, age=age, remark=remark,
                                       result=result,status=status)
    if res==0:
        return JsonResponse({'code': 400},safe=False)
    else:
        return JsonResponse({'code': 200},safe=False)
def docter_list(request):
    name = request.GET.get('name')
    sort = request.GET.get('sort')
    page = request.GET.get('page')
    limit = request.GET.get('limit')
    list = models.Userinfo.objects.all().filter(identity='2')
    if name:
        list = list.filter(username=name)
    if sort == '-id':
        list = list.order_by('-id')
    total = len(list)
    data = []
    for item in list:
        dict = {}
        dict['id'] = item.id
        dict['name'] = item.username
        dict['email'] = item.email
        data.append(dict)
    return JsonResponse({'code': 200, 'items': data, 'total': total}, safe=False)
@csrf_exempt
def update_byhospital(request):
    name = request.POST.get('name')
    email = request.POST.get('email')
    id = request.POST.get('id')
    #res为受影响的行数
    res=models.Userinfo.objects.filter(id=id).update(username=name,email=email)
    if res==0:
        return JsonResponse({'code': 400,'message':'修改失败'},safe=False)
    else:
        return JsonResponse({'code': 200},safe=False)
@csrf_exempt
def add_docter(request):
    name=request.POST.get("name")
    email=request.POST.get('email')
    id=models.Userinfo.objects.latest("id").id+1
    print(id)
    if name:
        try:
            res = models.Userinfo.objects.filter(username=name)
            if res.exists():
                return JsonResponse({'code': 400, 'message': '用户已存在'})
            else:
               userinfo=models.Userinfo(id=id,username=name,password='docter',identity='2',email=email)
               userinfo.save()
               return JsonResponse({'code':200,'message':'添加成功'},safe=False)
        except:
            return JsonResponse({'code':400,'message':'添加失败'},safe=False)
    else:
        return JsonResponse({'code': 400, 'message': '请填写用户名'}, safe=False)
def delete_docter(request):
    id=request.GET.get("id")
    try:
        res=models.Userinfo.objects.filter(id=id)
        res.delete()
        return JsonResponse({'code':200,'message':'删除成功'},safe=False)
    except:
        return JsonResponse({'code': 500, 'message': '删除失败'}, safe=False)

@csrf_exempt
def destroy(request):
    msg = {'code': 200}
    return JsonResponse(msg, safe=False)
@csrf_exempt
def exportpdf(request):
    id = request.POST.get("id")
    record = models.medicalrecord.objects.get(id=id)
    song = "simsun"
    pdfmetrics.registerFont(TTFont(song, "/home/www/project/hospital_2/hosptialapp/simsun.ttc"))
    PAGE_HEIGHT = A4[1]
    PAGE_WIDTH = A4[0]
    # 设置段落格式
    titleStyle = ParagraphStyle(
        name="titleStyle",
        alignment=1,
        fontName=song,
        fontSize=10,
        textColor=colors.black,
        backColor=HexColor(0xF2EEE9),
        borderPadding=(5, 5)
    )
    # 创建文档
    doc = SimpleDocTemplate("/home/andy/code/sys_collection/hospital_2/hosptialapp/output/pdftest.pdf")
    Story = [Spacer(1, 2 * inch)]
    # 保存文档
    doc.name = record.name
    gender=['男','女']
    doc.gender = gender[int(record.gender)-1]
    doc.age = record.age
    # doc.time = date = datetime.date.today().isoformat()
    doc.time=record.time
    doc.result = record.result
    doc.remark = record.remark
    doc.name_docter = 'Linda'
    doc.build(Story, onFirstPage=myFirstPage, onLaterPages=myLaterPages)
    webbrowser.open_new_tab('/home/www/project/hospital_2/hosptialapp/output/pdftest.pdf')
    return JsonResponse({'code':200,'path':'/home/www/project/hospital_2/hosptialapp/output/pdftest.pdf'}, safe=False)
class LogSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.log
        fields = '__all__'
@csrf_exempt
def loglist(request):
    list = models.log.objects.all().order_by("-id")
    total = len(list)
    serializer = LogSerializer(list, many=True)
    return JsonResponse({'code': 200, 'items': serializer.data, 'total': total}, safe=False)
@csrf_exempt
def edit_net(request):
    layer_no=request.POST.get('layer_no')
    kernel_no=request.POST.get('kernel_no')
    channel_no=request.POST.get('channel_no')
    num1=request.POST.get('num1')
    num2=request.POST.get('num2')
    input_no=request.POST.get('input_no')
    output_no=request.POST.get('output_no')
    weight=request.POST.get('weight')
    alter = models.alter
    try:
        alter_data = alter.objects.get(id=1)
        alter.objects.filter(id=1).update(id='1',layer_no=layer_no,kernel_no=kernel_no,channel_no=channel_no,num1=num1,num2=num2,input_no=input_no,weight=weight,output_no=output_no)
    except:
        alter.objects.create(id='1',layer_no=layer_no,kernel_no=kernel_no,channel_no=channel_no,num1=num1,num2=num2,input_no=input_no,weight=weight,output_no=output_no)

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
    res = {'conv1':w1.tolist(),'conv2':w2.tolist(),'fc':w4.tolist(),'code':200,'message':'修改成功'}
    print(layer_no)
    print(w4.tolist())
    print(w1.tolist())
    # if(layer_no<5):


    return JsonResponse(res,safe=False)
@csrf_exempt
def messyBlocks(request):
    path ='/home/www/project/UTCfiles/CSP/22/fileStore/'
    choiceRate = 1 / 100
    count = len(os.listdir(path))
    sn = math.floor(count * choiceRate)
    result = os.listdir(path)
    random.shuffle(result)
    result = result[:sn]
    print(result)  # 也可以改称返回数据已损坏的message
    for i in result:
        filePath = path + i
        f = open(filePath, 'r+')
        f.write("123")
    msg = {'code': 200}
    return JsonResponse(msg, safe=False)
@csrf_exempt
def reset_net(request):
    alter = models.alter
    alter.objects.filter(id=1).delete()
    return JsonResponse({'code':200,'msg':'重置成功'})
@csrf_exempt
def verify2(request):
    alter = models.alter
    layer_no=''
    kernel_no=''
    channel_no=''
    sleep(1)
    try:
        alter = models.alter
        alter=alter.objects.get(id=1)
        layer_no=alter.layer_no
        kernel_no=alter.kernel_no
        channel_no=alter.channel_no
        return JsonResponse({'code':200,'msg':'重置成功','layer_no':layer_no,'kernel_no':kernel_no,'channel_no':channel_no})
    except:
        return JsonResponse({'code':200,'msg':'重置成功','layer_no':'','kernel_no':'','channel_no':''})
@csrf_exempt
def destroy_location(request):
    uid=request.POST.get('uid')
    fid=request.POST.get('fid')
    checkError.checkEooro(fid,uid)
    with open('/home/www/project/hospital_2/UTC/checkResult.txt', 'r') as file:
        # 逐行读取文件内容，并将每行内容转换为整数，存储在列表中
        my_list = [int(line.strip()) for line in file]
    print(my_list)
    # error_node=my_list[-1]
    return JsonResponse({'code': 200,'error_node':my_list})
@csrf_exempt
def handledelete(request):
    id = request.POST.get('id')
    print(id)
    try:
        res = models.medicalrecord.objects.filter(id=id)
        res.delete()
        return JsonResponse({'code': 200, 'message': '删除成功'}, safe=False)
    except:
        return JsonResponse({'code': 500, 'message': '删除失败'}, safe=False)
@csrf_exempt
def handleFilter_multiattri(request):
    age_begin=request.POST.get('age_begin')
    age_end=request.POST.get('age_end')
    result_begin=request.POST.get('result_begin')
    result_end=request.POST.get('result_end')
    date_begin=datetime.strptime(request.POST.get('date_begin'),'%Y-%m-%dT%H:%M:%S.%fZ').timestamp()
    date_end=datetime.strptime(request.POST.get('date_end'),'%Y-%m-%dT%H:%M:%S.%fZ').timestamp()
    print(date_begin)
    print(date_end)
    # date_end=datetime.strptime(request.POST.get('date_end'),'%Y-%m-%d').timestamp()

    lower_bound = [age_begin,date_begin,result_begin]
    upper_bound = [age_end,date_end,result_end]
    msg = range_query(3,lower_bound,upper_bound)
    verify = range_query_verify()
    error_message = ''
    if len(verify[1])>1:
        error_message = verify[1][0]
    print('msg',msg)
    print(verify)
    # 【日期，年龄，推理结果】如果前端没有输入相应的值，在列表里对应为空
    # lower_bound = [date_begin, age_begin,'']
    # upper_bound = [date_end, age_end,'']
    # if result_begin!='' and result_end!='':
    #     lower_bound[2]=resultMap.index(result_begin)
    #     upper_bound[2]=resultMap.index(result_end)
    # dimension=len(lower_bound)
    # print(lower_bound)
    # print(upper_bound)
    # #TODO：多属性检索算法
    # #----开始------
    # list = models.medicalrecord.objects.all().order_by("-id")
    # total = len(list)
    # serializer = MedicalRecordSerializer(list, many=True)
    # # ----结束---
    #
    # # 检索后需要返回的值：items 为检索出来的记录，total为结果条数
    return JsonResponse({'code': 200, 'msg':msg, 'total': len(msg),'verify_result':verify[0],'verify_message':error_message}, safe=False)
@csrf_exempt
def handleFilter_keywords(request):
    # length=int(request.POST.get('len'))
    # keywords=[]
    # for i in range(0,length):
    #     my_str='keyword_'+ str(i)
    #     keywords.append(request.POST.get(my_str))
    keyword1=request.POST.get('keyword1')
    keyword2 = request.POST.get('keyword2')
    keyword3 = request.POST.get('keyword3')
    keyword4 = request.POST.get('keyword4')
    print('keyword4:',keyword4,type(keyword4))
    keywords=[keyword1,keyword2,keyword3,keyword4]
    print(keywords)
    result = keyword_query(keywords)
    patients = result[0]
    print('patients:',patients)
    verify_result = result[1]['result']
    error_message = result[1]['message']
    print(patients)
    # # TODO：多关键词检索算法
    # # ----开始------
    # list = models.medicalrecord.objects.all().order_by("-id")
    # total = len(list)
    # serializer = MedicalRecordSerializer(list, many=True)
    # ----结束---
    # 检索后需要返回的值：items 为检索出来的记录，total为结果条数
    return JsonResponse({'code': 200, 'msg': patients, 'total': len(patients),'verify_result':verify_result,'error_message':error_message}, safe=False)
@csrf_exempt
def createdata(request):
    data=request.POST.get('data')
    print(data)
    return JsonResponse({'code': 200}, safe=False)
@csrf_exempt
def update_keyword(request):#关键词检索更新
    id_before=request.POST.get('id_before')
    id_after=request.POST.get('id_after')
    print(id_before)
    print(id_after)
    key_alter(id_before,id_after)
    return JsonResponse({'code': 200}, safe=False)
import re
@csrf_exempt
def update_attri(request):#范围检索更新:id不修改
    id=request.POST.get('id')
    t=request.POST.get('time')
    time=str(datetime.strptime(t,'%Y-%m-%dT%H:%M:%S.%fZ'))
    age=request.POST.get('age')
    result=request.POST.get('result')
    print(id)
    print(time)
    print(age)
    print(result)
    range_alter(id,age,time,result)
    return JsonResponse({'code': 200}, safe=False)
@csrf_exempt
def verify_attri(request):#范围检索验证
    verify = range_query_verify()
    result = verify[0]
    print('verify',verify)
    error_message = ''
    if len(verify[1])>0:
        error_message = verify[1][0]
        print(error_message)
    # print('error_message2',verify[1],type(verify[1]))
    # print('error_message1', verify[1][0],type(verify[1][0]))
    print('verifyresult',result)
    print('error_message',error_message)
    return JsonResponse({'code': 200,'verify_result':result,'error_message':error_message}, safe=False)

@csrf_exempt
def verify_keyword(request):#关键词检索验证
    keyword1=request.POST.get('keyword1')
    keyword2 = request.POST.get('keyword2')
    keyword3 = request.POST.get('keyword3')
    keywords=[keyword1,keyword2,keyword3]
    result = key_verify(keywords)
    print('result',result['result'])
    print('message',result['message'])
    return JsonResponse({'code': 200,'verify_result':result['result'],'error_message':result['message']}, safe=False)
@csrf_exempt
def delete_arrti(request):#范围检索删除
    id = request.POST.get('id')
    delete_range(id)
    print(id)
    return JsonResponse({'code': 200}, safe=False)
@csrf_exempt
def delete_keyword(request):#关键词检索删除
    id = request.POST.get('id')
    key_delete(id)
    print(id)
    return JsonResponse({'code': 200}, safe=False)
@csrf_exempt
def infer_all(request):#推理全部
    id_begin=request.POST.get('id_begin')
    id_end=request.POST.get('id_end')
    print(id_begin)
    print(id_end)
    interfer_all()
    return JsonResponse({'code': 200}, safe=False)
@csrf_exempt
def infer_show(request):#推理全部
    id_begin=request.POST.get('id_begin')
    id_end=request.POST.get('id_end')
    print(id_begin)
    print(id_end)
    return JsonResponse({'code': 200,'data':[{'id':1,'result':'NO1'},{'id':2,'result':'NO1'}]}, safe=False)
@csrf_exempt
def write_log(request):
    id_length=int(request.POST.get('length'))
    request_list=request.POST
    log_info={}
    id_list=[]
    for i in range(0,id_length):
        id=request_list.get('ids[' + str(i) + ']')
        id_list.append(id)
        temp=[]
        for j in range(0,4):
            temp.append(request_list.get('plaintext[' + str(i) + ']['+str(j)+']'))
        log_info[id]=temp
    print('idlist',id_list)
    plain_interfer = compare(id_list)
    plain_interfer_message = {}

    for i in range(len(plain_interfer)):
        plain_interfer_message.update({str(id_list[i]):plain_interfer[i]})
    print('idlist',id_list)
    print('密态推理后的解密结果：',log_info)
    print('明文推理的结果：',plain_interfer_message)
    return JsonResponse({'code': 200}, safe=False)