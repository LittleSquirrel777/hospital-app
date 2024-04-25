import socket
import time
import struct
import json
import numpy as np
import socket
import sys
import pickle
headerSize = 12
host = '10.201.153.204'
port = 1234

ADDR = (host, port)

if __name__ == '__main__':
    header = [1,2,3]
    headPack = struct.pack("!3I", *header)
    print()
    client = socket.socket()
    client.connect(ADDR)
    pass

def send(connecting_socket,data,ver=1,cmd=101):

    data = json.dumps(data)

    # 加上应用层协议头部，封装应用层数据
    body = data
    print(body)
    header = [ver, body.__len__(), cmd]
    headPack = struct.pack("!3I", *header)
    sendData1 = headPack+body.encode()

    # # 分包测试
    # ver = 2
    # body = json.dumps(dict(hello="world2"))
    # print(body)
    # cmd = 102
    # header = [ver, body.__len__(), cmd]
    # headPack = struct.pack("!3I", *header)
    # sendData2_1 = headPack+body[:2].encode()
    # sendData2_2 = body[2:].encode()
    #
    # # 粘包测试
    # ver = 3
    # body1 = json.dumps(dict(hello="world3"))
    # print(body1)
    # cmd = 103
    # header = [ver, body1.__len__(), cmd]
    # headPack1 = struct.pack("!3I", *header)
    #
    # ver = 4
    # body2 = json.dumps(dict(hello="world4"))
    # print(body2)
    # cmd = 104
    # header = [ver, body2.__len__(), cmd]
    # headPack2 = struct.pack("!3I", *header)
    #
    # sendData3 = headPack1+body1.encode()+headPack2+body2.encode()


    # 正常数据包
    connecting_socket.send(sendData1)
    time.sleep(3)

    # # 分包测试
    # client.send(sendData2_1)
    # time.sleep(0.2)
    # client.send(sendData2_2)
    # time.sleep(3)
    #
    # # 粘包测试
    # client.send(sendData3)
    # time.sleep(3)
    # client.close()


def message_handle(client,addr):
    """
    消息处理
    """
    dataBuffer = bytes()
    sn = 0
    while True:
        data = client.recv(1024)
        if data:
            # 把数据存入缓冲区
            dataBuffer += data
            while True:
                if len(dataBuffer) < headerSize:
                    print("数据包（%s Byte）小于包头长度，跳出小循环" % len(dataBuffer))
                    break

                # 读取包头
                # struct中:!代表Network order，3I代表3个unsigned int数据
                headPack = struct.unpack('!3I', dataBuffer[:headerSize])
                bodySize = headPack[1]

                # 分包情况处理，跳出函数继续接收数据
                if len(dataBuffer) < headerSize + bodySize:
                    print("数据包（%s Byte）不完整（总共%s Byte），等待后续数据包到达" % (len(dataBuffer), headerSize + bodySize))
                    break
                # 读取包体的内容
                body = dataBuffer[headerSize:headerSize + bodySize]

                res = dataHandle(headPack,body,addr,client,sn)

                # 粘包情况的处理
                dataBuffer = dataBuffer[headerSize + bodySize:]

                return res
        else:
            break


def dataHandle(headPack,body,addr,client,sn):
    sn += 1
    print(addr,"返回的第%s个数据包" % sn)


    if (headPack[2]==202):
        keys_en = json.loads(body.decode())
        print("这是经过AES加密过后的Paillier公私钥",keys_en)
        send(client,"公私钥已收到，可以开始联邦学习",1,203)
        return keys_en


