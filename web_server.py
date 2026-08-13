from flask import Flask, Response, request, jsonify
from flask_cors import CORS

import cv2
import threading
import time


app = Flask(__name__)

CORS(app)


# =========================
# 全局视频帧
# =========================

latest_frame = None

frame_lock = threading.Lock()


# =========================
# 当前 YOLO 检测类别
# =========================

current_classes = [
    "person",
    "car",
    "dog",
    "cup",
    "phone"
]


# =========================
# 【新增】前端显示类别（用于主循环筛选框）
# =========================

_current_display_classes = set()

def set_display_classes(classes):
    """从前端接收需要显示的类别列表"""
    global _current_display_classes
    _current_display_classes = set(classes)

def get_display_classes():
    """主循环调用，获取当前需要显示的类别"""
    return _current_display_classes


# =========================
# YOLO 调用
# =========================

def update_frame(frame):

    global latest_frame

    if frame is None:
        return

    with frame_lock:
        latest_frame = frame.copy()


# =========================
# 视频流
# =========================

def generate():

    global latest_frame

    while True:

        # 取最新帧
        with frame_lock:

            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        # 还没有帧
        if frame is None:

            time.sleep(0.01)

            continue

        # 调整大小
        frame = cv2.resize(
            frame,
            (640, 360)
        )

        # JPEG 编码
        ret, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80
            ]
        )

        if not ret:

            time.sleep(0.01)

            continue

        # MJPEG
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: "
            + str(len(jpeg)).encode()
            + b"\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )

        time.sleep(0.01)


# =========================
# 视频接口
# =========================

@app.route("/video")
def video():

    return Response(
        generate(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# =========================
# 首页
# =========================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>YOLOE Front Camera</title>
    </head>

    <body>

        <h2>YOLOE Front Camera</h2>

        <img
            src="/video"
            width="640"
            height="360"
        >

    </body>
    </html>
    """


# =========================
# 修改检测类别
# =========================

@app.route(
    "/set_classes",
    methods=["POST"]
)
def set_classes():

    global current_classes

    data = request.get_json(
        silent=True
    )

    if not data:

        return jsonify({
            "status": "error",
            "msg": "没有收到JSON数据"
        }), 400

    classes = data.get(
        "classes",
        []
    )

    if len(classes) == 0:

        return jsonify({
            "status": "error",
            "msg": "至少选择一个目标"
        }), 400

    current_classes = classes.copy()
    
    # ========== 【新增】同步更新前端显示类别 ==========
    set_display_classes(current_classes)

    print(
        "新的识别目标:",
        current_classes
    )

    return jsonify({
        "status": "ok",
        "classes": current_classes
    })


# =========================
# 启动
# =========================

def start_web():

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False,
        use_reloader=False
    )


if __name__ == "__main__":

    start_web()
