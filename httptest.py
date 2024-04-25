import requests
import json
import time

def send(data,session):
    # time.sleep(1)
    url = "http://127.0.0.1:1316/post"
    # print(data)
    data = json.dumps(data)
    session.keep_alive = False
    res = session.post(url=url, data=data,
                        headers={"Content-Type": "application/json", "Connection":"close"},verify=False)  # 这里传入的data,是body里面的数据。params是拼接url时的参数
    # print("发送的body:", res.request.body)
    # print("response返回结果：", res.json())
    print('')
    return res.json()


if __name__=='__main__':
    url = "http://127.0.0.1:1316/post"
    myParams = {"key": "username", "info": "plusroax"}  # 字典格式，推荐使用，它会自动帮你按照k-v拼接url
    data = {"name": "plusroax","age": 18} # Post请求发送的数据，字典格式
    data = json.dumps(data)
    res = requests.post(url=url, data = data,headers={'Content-Type': "application/json"})#这里传入的data,是body里面的数据。params是拼接url时的参数
    # res = requests.get(url=url)

    print("发送的body:",res.request.body)
    print(res)
    # print("response返回结果：",res.json())

    # print('url:', res.request.url)  # 查看发送的url
    # print("response:", res.text)  #
