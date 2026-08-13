from flask import Flask, Response
from flask import request, jsonify
from flask_cors import CORS

import cv2
import threading
import time  # ← 新增导入


app = Flask(__name__)

CORS(app)

latest_frame = None
frame_lock = threading.Lock()

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


def update_frame(frame):
    """
    给YOLO脚本调用
    """
    global latest_frame

    with frame_lock:
        latest_frame = frame.copy()


def generate():
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)  # ← 修复：避免死循环
                continue
            frame = latest_frame.copy()

        frame = cv2.resize(frame, (640, 360))

        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        if not ret:
            time.sleep(0.01)  # ← 修复：避免死循环
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type:image/jpeg\r\n\r\n"
            +
            jpeg.tobytes()
            +
            b"\r\n"
        )
        
        time.sleep(0.03)  # ← 限制帧率约30fps


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    return """
    <html>
    <body>
    <h2>YOLOE Rear Camera</h2>
    <img src="/video">
    </body>
    </html>
    """


@app.route("/set_classes", methods=["POST"])
def set_classes():
    global current_classes

    data = request.json
    classes = data.get("classes", [])

    if len(classes) == 0:
        return jsonify({
            "status": "error",
            "msg": "至少选择一个目标"
        })

    current_classes = classes.copy()
    
    # ========== 【新增】同步更新前端显示类别 ==========
    set_display_classes(current_classes)

    print("新的识别目标（子码流）:", current_classes)

    return jsonify({
        "status": "ok",
        "classes": current_classes
    })


def start_web():
    app.run(
        host="0.0.0.0",
        port=5001,  # 端口改为5001
        threaded=True
    )