from flask import Flask, Response, request, jsonify
from flask_cors import CORS

import cv2
import threading
import time


app = Flask(__name__)

CORS(app)

no_alert_objects_global = []
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


import json

# 在 API_BASE_2 对应的 Flask 后端中添加

# 全局变量存储接收到的告警设置
received_audio_config = {'classes': [], 'language': []}

@app.route('/api/audio_alert_settings', methods=['POST'])
def receive_audio_alert_settings():
    """接收前端发送的告警设置"""
    global received_audio_config
    global current_classes  # 引用全局变量
    global no_alert_objects_global  # 使用这个
    global _current_display_classes  # 添加这行


    try:
        data = request.get_json()
        
        # 接收 classes 和 language
        received_audio_config['classes'] = data.get('classes', [])
        received_audio_config['language'] = data.get('language', [])
        
        # ===== 合并到 current_classes =====
        with classes_lock:
            # 获取当前无告警对象（从 new_classes 或 silent_alert_objects）
            # 假设无告警对象存储在 silent_alert_objects 中
            no_alert_objects = no_alert_objects_global  # 或者从其他地方获取
            
            # 取并集：无告警对象 + 告警对象
            merged = list(set(no_alert_objects + received_audio_config['classes']))
            current_classes = merged.copy()
            
            # 同步更新 _current_display_classes
            _current_display_classes = set(current_classes)
        # ==================================
        
        print(f"[API_BASE_2] 收到告警设置:")
        print(f"[API_BASE_2] classes: {received_audio_config['classes']}")
        print(f"[API_BASE_2] language: {received_audio_config['language']}")
        print(f"[API_BASE_2] 无告警对象: {no_alert_objects}")
        print(f"[API_BASE_2] 合并后 current_classes: {current_classes}")
        
        return jsonify({
            'status': 'success',
            'classes': received_audio_config['classes'],
            'language': received_audio_config['language']
        })
        
    except Exception as e:
        print(f"[API_BASE_2] 接收失败: {e}")
        return jsonify({'status': 'error', 'msg': str(e)}), 500

    
# ============================================================
# 前端修改 YOLO 检测类别
# ============================================================

@app.route(
    "/set_classes",
    methods=["POST"]
)
def set_classes_api():
    global current_classes
    global no_alert_objects_global
    global _current_display_classes  # 添加这行
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

    # 保存无告警对象到全局变量
    no_alert_objects_global = new_classes.copy()

# ===== 核心逻辑：合并无告警对象和告警对象 =====
    with classes_lock:
        # 获取当前告警对象
        alert_objects = received_audio_config.get('classes', [])
        
        if new_classes:  # 接收到非空列表：无告警对象 + 告警对象
            merged = list(set(new_classes + alert_objects))
            current_classes = merged.copy()
        else:  # 接收到空列表：只保留告警对象
            current_classes = alert_objects.copy()
        
        # 同步更新 _current_display_classes
        _current_display_classes = set(current_classes)

    # 控制台打印
    print()
    print("========================================")
    print("后摄像头收到新的 YOLO 检测类别")
    if new_classes:
        print(f"无告警对象: {new_classes}")
        print(f"告警对象: {alert_objects}")
        print(f"合并后: {current_classes}")
    else:
        print("收到空列表，清除无告警对象")
        print(f"只保留告警对象: {alert_objects}")
    print("========================================")
    print()

    # 返回给前端
    return jsonify({
        "status": "success",
        "msg": "检测类别更新成功",
        "classes": current_classes
    })





# web_server.py 中需要有这两个路由

@app.route('/api/get_audio_alert_config', methods=['GET'])
def get_audio_alert_config_api():
    """返回当前告警配置"""
    global received_audio_config
    return jsonify(received_audio_config)

@app.route('/api/update_detected_indices', methods=['POST'])
def update_detected_indices():
    """接收YOLO推送的检测索引"""
    global detected_alert_indices
    data = request.get_json()
    detected_alert_indices = data.get('indices', [])
    print(f"[UPDATE] 收到检测索引: {detected_alert_indices}")
    return jsonify({'status': 'success'})


  # ===== 新增：前端轮询获取索引的接口 =====
@app.route('/api/get_detected_alert_indices', methods=['GET'])
def get_detected_alert_indices_api():
    """返回当前检测到的告警对象索引列表"""
    global detected_alert_indices
    
    indices = detected_alert_indices.copy()
    
    print(f"[GET] 返回检测索引: {indices}")
    
    return jsonify(indices)  # 直接返回列表
    


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