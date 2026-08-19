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
# YOLO 当前检测类别
# ============================================================

current_classes = [
    ""
]

classes_lock = threading.Lock()


def get_current_classes():
    """获取当前 YOLO 检测类别"""
    global current_classes
    with classes_lock:
        return current_classes.copy()


# ============================================================
# 前端需要显示 / 框选的类别
# ============================================================

_current_display_classes = set()

display_classes_lock = threading.Lock()


def set_display_classes(classes):
    """
    设置前端要求显示的类别

    例如：
        ["person", "chair", "dog"]

    保存为：
        {"person", "chair", "dog"}
    """

    global _current_display_classes

    with display_classes_lock:

        _current_display_classes = set(classes)


def get_display_classes():
    """
    YOLO 主循环调用

    返回：
        set
    """

    with display_classes_lock:

        return set(_current_display_classes)


# ============================================================
# 前端传入的"不报警对象"
# ============================================================

_no_alert_classes = set()

no_alert_lock = threading.Lock()


def set_no_alert_classes(classes):
    """
    设置不报警类别
    """

    global _no_alert_classes

    with no_alert_lock:

        _no_alert_classes = set(classes)


def get_no_alert_classes():
    """
    YOLO 主程序调用

    获取当前前端设置的不报警类别
    """

    with no_alert_lock:

        return set(_no_alert_classes)


# ============================================================
# YOLO 更新视频帧
# ============================================================

def update_frame(frame):

    global latest_frame

    if frame is None:
        return

    with frame_lock:

        latest_frame = frame.copy()

def generate():

    global latest_frame

    while True:

        # ----------------------------------------
        # 获取最新帧
        # ----------------------------------------

        with frame_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()


        # ----------------------------------------
        # 没有视频帧
        # ----------------------------------------

        if frame is None:

            time.sleep(0.01)

            continue


        # ----------------------------------------
        # 调整视频大小
        # ----------------------------------------

        frame = cv2.resize(
            frame,
            (640, 360)
        )


        # ----------------------------------------
        # JPEG 编码
        # ----------------------------------------

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


        # ----------------------------------------
        # MJPEG
        # ----------------------------------------

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
# 获取当前 YOLO 检测类别
# ============================================================

@app.route(
    "/get_classes",
    methods=["GET"]
)
def get_classes_api():

    with classes_lock:

        classes = current_classes.copy()


    return jsonify({

        "status": "ok",

        "classes": classes

    })


# ============================================================
# 接收 JS 的 objects（核心接口）
#
# JS 发送：
# {
#     "objects": ["person", "chair", "dog"]
# }
#
# 或重置时：
# {
#     "objects": ["#"]
# }
# ============================================================

@app.route(
    "/set_no_alert_objects",
    methods=["POST"]
)
def set_no_alert_objects_api():
    global current_classes
    global silent_alert_objects

    global current_classes

    data = request.get_json(
        silent=True
    )

    if data is None:
        return jsonify({
            "status": "error",
            "msg": "没有收到JSON数据"
        }), 400

    objects = data.get(
        "objects",
        []
    )

    if not isinstance(objects, list):

        return jsonify({
            "status": "error",
            "msg": "objects 必须是列表"
        }), 400

    # ================================================
    # 检查是否是重置操作（包含 "#"）
    # ================================================
    if "#" in objects:
    
    # 清空不报警类别
        set_no_alert_classes([])
        silent_alert_objects = []
    # 重置：只保留声音告警对象，清除无声告警对象
        with classes_lock:
            # 获取当前声音告警对象
            audio_objects = audio_alert_config.get('classes', [])
            # 只保留声音告警对象
            current_classes = audio_objects.copy()
        
        print()
        print("==============================")
        print("[WEB] 执行重置操作")
        print("[WEB] 已清空不报警类别")
        print("[WEB] 保留声音告警对象:", audio_objects)
        print("[WEB] 当前检测类别:", current_classes)
        print("==============================")
        print()

        return jsonify({
            "status": "ok",
            "msg": "已重置",
            "objects": [],
            "classes": current_classes
        })

    # ================================================
    # 正常设置不报警对象
    # ================================================
    clean_objects = []

    for obj in objects:

        if not isinstance(obj, str):
            continue

        obj = obj.strip()

        if obj:
            clean_objects.append(obj)

    # 去重
    clean_objects = list(
        dict.fromkeys(clean_objects)
    )

    # 设置不报警类别
    set_no_alert_classes(
        clean_objects
    )
    silent_alert_objects = clean_objects.copy()  # 保存无声告警对象
    # ★ 传给 YOLOE 作为检测类别
    with classes_lock:

        # 获取当前声音告警对象
        audio_objects = audio_alert_config.get('classes', [])
        # 合并：声音告警对象 + 无声告警对象
        current_classes = list(set(audio_objects + clean_objects))

    print()
    print("==============================")
    print("[WEB] 收到 Prompt:")
    print(clean_objects)

    print("[WEB] 当前 YOLOE 类别:")
    print(current_classes)

    print("==============================")
    print()

    return jsonify({

        "status": "ok",

        "objects": clean_objects,

        "classes": clean_objects
   })

# ============================================================
# 获取当前不报警类别
# ============================================================

@app.route(
    "/get_no_alert_objects",
    methods=["GET"]
)
def get_no_alert_objects_api():

    objects = list(
        get_no_alert_classes()
    )


    return jsonify({

        "status": "ok",

        "objects": objects

    })

import json

# 全局变量存储声音告警设置
# 全局变量
audio_alert_config = {'classes': []}
silent_alert_objects = []  # 新增：单独存储无声告警对象

@app.route('/api/audio_alert_settings', methods=['POST'])
def save_audio_alert_settings():
    try:
        global audio_alert_config
        global current_classes
        global silent_alert_objects  # 引用全局变量
        
        data = request.get_json()
        audio_alert_config['classes'] = data.get('classes', [])
        audio_alert_config['language'] = data.get('language', [])  # 现在是数组
        # 重置逻辑：current_classes = 无声告警对象 + 新的声音告警对象
        with classes_lock:
            # 直接使用 silent_alert_objects 作为无声告警对象
            merged = list(set(silent_alert_objects + audio_alert_config['classes']))
            current_classes = merged.copy()
        
        print(f"[AUDIO ALERT] 声音告警对象: {audio_alert_config['classes']}")
        print(f"[SILENT] 无声告警对象: {silent_alert_objects}")
        print(f"[MERGED] 合并后检测类别: {current_classes}")
        
        return jsonify({'status': 'success', 'classes': audio_alert_config['classes']})
    
    except Exception as e:
        import traceback
        print(f"[ERROR] save_audio_alert_settings 异常: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'msg': str(e)}), 500


# web_server.py 中添加
@app.route('/api/get_audio_alert_config', methods=['GET'])
def get_audio_alert_config_api():
    """返回当前声音告警配置"""
    global audio_alert_config
    print(f"[WEB] 返回告警配置: {audio_alert_config}")  # 添加调试
    return jsonify(audio_alert_config)


detected_alert_indices = []

@app.route('/api/update_detected_indices', methods=['POST'])
def update_detected_indices():
    """YOLO 主程序调用，更新检测到的索引"""
    global detected_alert_indices
    data = request.get_json()
    detected_alert_indices = data.get('indices', [])
    print(f"[WEB SERVER] 收到检测到的索引: {detected_alert_indices}")  # 添加这行
    return jsonify({'status': 'success'})

detected_indices_lock = threading.Lock()  # 添加这行
@app.route('/api/get_detected_alert_indices', methods=['GET'])
def get_detected_alert_indices_api():
    """返回当前检测到的告警对象索引列表"""
    global detected_alert_indices
    global audio_alert_config
    
    with detected_indices_lock:
        indices = detected_alert_indices.copy()
    
    # language 现在是数组
    language = audio_alert_config.get('language', [])
    
    return jsonify({
        'indices': indices,
        'objects': [audio_alert_config.get('classes', [])[i] if i < len(audio_alert_config.get('classes', [])) else '' for i in indices],
        'language': language
    })
# ============================================================
# 获取当前显示类别
# ============================================================

@app.route(
    "/get_display_classes",
    methods=["GET"]
)
def get_display_classes_api():

    classes = list(
        get_display_classes()
    )


    return jsonify({

        "status": "ok",

        "classes": classes

    })


# ============================================================
# 启动 Flask
# ============================================================

def start_web():

    app.run(

        host="0.0.0.0",

        port=5000,

        threaded=True,

        debug=False,

        use_reloader=False

    )


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    start_web()