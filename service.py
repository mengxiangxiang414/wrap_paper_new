import base64
from flask import request
from flask import Flask
import os
import cv2
import time
import json
from main_one_photo_LSD_server import process

app = Flask(__name__)


# 定义路由
@app.route("/photo", methods=['POST'])
def get_frame():
    # 接收图片
    upload_file = request.files['file']
    # 获取图片名
    file_name = upload_file.filename

    # 文件保存目录（桌面）
    file_path = r'data'
    if upload_file:
        # 地址拼接
        file_paths = os.path.join(file_path, file_name)
        # 保存接收的图片到桌面
        upload_file.save(file_paths)
        # 随便打开一张其他图片作为结果返回，
        time1 = time.time()
        try:
            process(file_paths)
            with open(file_paths, 'rb') as f:
                res = base64.b64encode(f.read())
                return res
        except:
            return 'false'


if __name__ == "__main__":
    app.run()