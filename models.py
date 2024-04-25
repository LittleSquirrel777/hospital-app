import django
from django.db import models
import datetime
from rest_framework import serializers
from django.utils import timezone


# Create your models here.
class Userinfo(models.Model):
    id_choice=(
        ('1','患者'),
        ('2','医生'),
        ('3','医院')
    )
    id=models.IntegerField(primary_key=True)
    username=models.CharField(max_length=45)
    password=models.CharField(max_length=45)
    identity=models.CharField(choices=id_choice,max_length=2)
    email=models.CharField(max_length=45)
    class Meta:
        db_table='Userinfo'
class medical_record(models.Model):
    gender_choice=(
        ('1','男'),
        ('2','女')
    )
    id=models.IntegerField(primary_key=True)
    name=models.CharField(max_length=45)
    gender=models.CharField(choices=gender_choice,max_length=2)
    age=models.IntegerField()
    desc=models.CharField(max_length=45)
    # time=models.DateField()
    otherdata=models.CharField(max_length=45)
    class Meta:
        db_table='medical_record'
class medicalrecord(models.Model):
    gender_choice = (
        ('1', '男'),
        ('2', '女')
    )
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=45)
    gender = models.CharField(choices=gender_choice, max_length=2)
    age = models.IntegerField()
    result = models.CharField(max_length=45)
    remark = models.CharField(max_length=45)
    time = models.DateField(default=timezone.now)
    status = models.CharField(max_length=45)
    feature1 = models.CharField(max_length=45)
    feature2 = models.CharField(max_length=45)
    verify_res = models.CharField(max_length=45)
    verify_error_layer = models.CharField(max_length=200)
    verify_error_kernel = models.CharField(max_length=255)

    class Meta:
        db_table = 'medicalrecord'
class log(models.Model):
    id=models.IntegerField(primary_key=True)
    time=models.DateField()
    record_id=models.IntegerField()
    error_num=models.IntegerField()
    error_node=models.CharField(max_length=45)
    class Meta:
        db_table='log'
class alter(models.Model):


    # gender_choice=(
    #     ('1','男'),
    #     ('2','女')
    # )
    id=models.CharField(primary_key=True,max_length=45)
    layer_no=models.CharField(max_length=45)
    kernel_no=models.CharField(max_length=45)
    channel_no=models.CharField(max_length=45)
    num1=models.CharField(max_length=45)
    num2=models.CharField(max_length=45)
    input_no=models.CharField(max_length=45)
    output_no=models.CharField(max_length=45)
    weight=models.CharField(max_length=45)
    class Meta:
        db_table='alter'
