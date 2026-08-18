from flask import Flask, Response, request, jsonify
from flask_cors import CORS

import cv2
import threading
import time


app = Flask(__name__)

CORS(app)


# ============================================================
# 全局视频帧
# ============================================================

latest_frame = None

frame_lock = threading.Lock()


# ============================================================
# 当前 YOLO 检测类别
# ============================================================

current_classes = [
    ""
]

classes_lock = threading.Lock()


# ============================================================
# 当前前端要求的检测类别
# ============================================================


def set_display_classes(classes):
    """
    设置当前需要检测的类别
    """
    global _current_display_classes

    with classes_lock:
        _current_display_classes = set(classes)




_current_display_classes = set(current_classes)



def get_display_classes():
    """
    YOLO 主程序调用

    返回当前前端设置的检测类别
    """

    with classes_lock:
        return list(_current_display_classes)


# ============================================================
# YOLO 主程序更新视频帧
# ============================================================

def update_frame(frame):
    """
    给 YOLO 主程序调用
    将 YOLO 处理后的画面传给 Flask
    """

    global latest_frame

    if frame is None:
        return

    with frame_lock:
        latest_frame = frame.copy()


# ============================================================
# 视频流
# ============================================================

def generate():

    global latest_frame

    while True:

        # -----------------------------
        # 获取最新帧
        # -----------------------------

        with frame_lock:

            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()


        # -----------------------------
        # 没有视频帧
        # -----------------------------

        if frame is None:

            time.sleep(0.01)

            continue


        # -----------------------------
        # 调整视频大小
        # -----------------------------

        frame = cv2.resize(
            frame,
            (640, 360)
        )


        # -----------------------------
        # JPEG编码
        # -----------------------------

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


        # -----------------------------
        # MJPEG视频流
        # -----------------------------

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: "
            + str(len(jpeg)).encode()
            + b"\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )


        # -----------------------------
        # 限制视频流速度
        # -----------------------------

        time.sleep(0.03)


# ============================================================
# 视频接口
# ============================================================

@app.route("/video")
def video():

    return Response(
        generate(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# 前端修改 YOLO 检测类别
# ============================================================

@app.route(
    "/set_classes",
    methods=["POST"]
)
def set_classes_api():
    global current_classes
    
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "status": "error",
            "msg": "没有收到JSON数据"
        }), 400

    classes = data.get("objects", [])

    if not isinstance(classes, list):
        return jsonify({
            "status": "error",
            "msg": "classes 必须是列表"
        }), 400

    # 清理类别名称
    new_classes = []
    for cls in classes:
        cls = str(cls).strip()
        if cls:
            new_classes.append(cls)

    # if len(new_classes) == 0:
    #     return jsonify({
    #         "status": "error",
    #         "msg": "至少输入一个有效的检测目标"
    #     }), 400

    # ========================================================
    # 关键：将前端获取的 classes 赋值给 get_display_classes
    # 调用 set_display_classes 函数，将值存入 _current_display_classes
    # ========================================================
    set_display_classes(new_classes)  # 这行就是将前端classes赋值给get_display_classes

    # 同时也更新 current_classes（如果需要的话）
    with classes_lock:
        current_classes = new_classes.copy()

    # 控制台打印
    print()
    print("========================================")
    print("后摄像头收到新的 YOLO 检测类别")
    print("检测类别：", current_classes)
    print("========================================")
    print()

    # 返回给前端
    return jsonify({
        "status": "success",
        "msg": "检测类别更新成功",
        "classes": current_classes
    })

    


    # ========================================================
    # 获取 JSON
    # ========================================================

    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({

            "success": False,

            "message": "没有收到JSON数据"

        }), 400


    # ========================================================
    # 获取 classes
    # ========================================================

    classes = data.get(
        "classes",
        []
    )


    # ========================================================
    # 检查 classes 是否为列表
    # ========================================================

    if not isinstance(classes, list):

        return jsonify({

            "success": False,

            "message": "classes必须是列表"

        }), 400


    # ========================================================
    # 清理类别名称
    #
    # 例如：
    #
    # ["person", " dog ", "", "cat"]
    #
    # 变成：
    #
    # ["person", "dog", "cat"]
    # ========================================================

    new_classes = []

    for cls in classes:

        cls = str(cls).strip()

        if cls:

            new_classes.append(cls)

    # ========================================================
    # 更新当前检测类别
    # ========================================================

    with classes_lock:

        current_classes = new_classes.copy()
    # ========================================================
    # 控制台打印
    # ========================================================

    print()
    print("========================================")
    print("后摄像头收到新的 YOLO 检测类别")
    print("检测类别：", current_classes)
    print("========================================")
    print()


    # ========================================================
    # 返回给前端
    # ========================================================

    return jsonify({

        "success": True,

        "message": "检测类别更新成功",

        "classes": current_classes

    })


# ============================================================
# 启动 Flask
# ============================================================

def start_web():

    app.run(

        host="0.0.0.0",

        port=5001,

        threaded=True,

        debug=False,

        use_reloader=False

    )


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    start_web()