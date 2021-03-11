import requests
import base64

# API地址
url = "http://127.0.0.1:5000/photo"
# 图片地址
file_path = 'data/5.jpg'
# 图片名
file_name = file_path.split('/')[-1]
# 二进制打开图片
file = open(file_path, 'rb')
# 拼接参数
files = {'file': (file_name, file, 'image/jpg')}
# 发送post请求到服务器端
r = requests.post(url, files=files)
# 获取服务器返回的图片，字节流返回
result = r.content
# 字节转换成图片
img = base64.b64decode(result)
file = open('H:/out/img_5.jpg', 'wb')
file.write(img)
file.close()