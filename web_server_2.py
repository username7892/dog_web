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
# SMS 配置
# ============================================================

SMS_CONFIG = {
    "alert_object": None,
    "contact_phone": None
}


# ============================================================
# YOLO 当前检测类别
# ============================================================

current_classes = [""]
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
    global _current_display_classes
    with display_classes_lock:
        _current_display_classes = set(classes)


def get_display_classes():
    with display_classes_lock:
        return set(_current_display_classes)


# ============================================================
# 前端传入的“不报警对象”
# ============================================================

_no_alert_classes = set()
no_alert_lock = threading.Lock()


def set_no_alert_classes(classes):
    global _no_alert_classes
    with no_alert_lock:
        _no_alert_classes = set(classes)


def get_no_alert_classes():
    with no_alert_lock:
        return set(_no_alert_classes)


# ============================================================
# 声音告警配置 & 无声告警对象
# ============================================================

audio_alert_config = {'classes': []}
silent_alert_objects = []


# ============================================================
# YOLO 更新视频帧
# ============================================================

def update_frame(frame):
    global latest_frame
    if frame is None:
        return
    with frame_lock:
        latest_frame = frame.copy()


# ============================================================
# MJPEG 视频流
# ============================================================

def generate():
    global latest_frame
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (640, 360))

        ret, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not ret:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: "
            + str(len(jpeg)).encode()
            + b"\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )

        time.sleep(0.03)


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# 获取当前 YOLO 检测类别
# ============================================================

@app.route("/get_classes", methods=["GET"])
def get_classes_api():
    with classes_lock:
        classes = current_classes.copy()
    return jsonify({
        "status": "ok",
        "classes": classes
    })


# ============================================================
# 设置不报警对象（核心接口）
# ============================================================

@app.route("/set_no_alert_objects", methods=["POST"])
def set_no_alert_objects_api():
    global current_classes
    global silent_alert_objects

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "status": "error",
            "msg": "没有收到JSON数据"
        }), 400

    objects = data.get("objects", [])
    if not isinstance(objects, list):
        return jsonify({
            "status": "error",
            "msg": "objects 必须是列表"
        }), 400

    # ================= 重置 =================
    if "#" in objects:
        set_no_alert_classes([])
        silent_alert_objects = []

        with classes_lock:
            audio_objects = audio_alert_config.get('classes', [])
            current_classes = audio_objects.copy()

        print("\n==============================")
        print("[WEB] 执行重置操作")
        print("[WEB] 已清空不报警类别")
        print("[WEB] 保留声音告警对象:", audio_objects)
        print("[WEB] 当前检测类别:", current_classes)
        print("==============================\n")

        return jsonify({
            "status": "ok",
            "msg": "已重置",
            "objects": [],
            "classes": current_classes
        })

    # ================= 正常设置 =================
    clean_objects = []
    for obj in objects:
        if not isinstance(obj, str):
            continue
        obj = obj.strip()
        if obj:
            clean_objects.append(obj)

    clean_objects = list(dict.fromkeys(clean_objects))

    set_no_alert_classes(clean_objects)
    silent_alert_objects = clean_objects.copy()

    with classes_lock:
        audio_objects = audio_alert_config.get('classes', [])
        current_classes = list(set(audio_objects + clean_objects))

    print("\n==============================")
    print("[WEB] 收到 Prompt:")
    print(clean_objects)
    print("[WEB] 当前 YOLO 类别:")
    print(current_classes)
    print("==============================\n")

    return jsonify({
        "status": "ok",
        "objects": clean_objects,
        "classes": clean_objects
    })


# ============================================================
# 获取不报警类别
# ============================================================

@app.route("/get_no_alert_objects", methods=["GET"])
def get_no_alert_objects_api():
    objects = list(get_no_alert_classes())
    return jsonify({
        "status": "ok",
        "objects": objects
    })


# ============================================================
# 声音告警设置
# ============================================================

@app.route('/api/audio_alert_settings', methods=['POST'])
def save_audio_alert_settings():
    global audio_alert_config
    global current_classes
    global silent_alert_objects

    try:
        data = request.get_json()
        audio_alert_config['classes'] = data.get('classes', [])
        audio_alert_config['language'] = data.get('language', [])

        with classes_lock:
            merged = list(set(silent_alert_objects + audio_alert_config['classes']))
            current_classes = merged.copy()

        print(f"[AUDIO ALERT] 声音告警对象: {audio_alert_config['classes']}")
        print(f"[SILENT] 无声告警对象: {silent_alert_objects}")
        print(f"[MERGED] 合并后检测类别: {current_classes}")

        return jsonify({
            'status': 'success',
            'classes': audio_alert_config['classes']
        })

    except Exception as e:
        import traceback
        print(f"[ERROR] save_audio_alert_settings 异常: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'msg': str(e)}), 500


@app.route('/api/get_audio_alert_config', methods=['GET'])
def get_audio_alert_config_api():
    return jsonify(audio_alert_config)


# ============================================================
# 检测到的告警索引
# ============================================================

detected_alert_indices = []
detected_indices_lock = threading.Lock()


@app.route('/api/update_detected_indices', methods=['POST'])
def update_detected_indices():
    global detected_alert_indices
    data = request.get_json()
    detected_alert_indices = data.get('indices', [])
    print(f"[WEB SERVER] 收到检测到的索引: {detected_alert_indices}")
    return jsonify({'status': 'success'})


@app.route('/api/get_detected_alert_indices', methods=['GET'])
def get_detected_alert_indices_api():
    global detected_alert_indices
    global audio_alert_config

    with detected_indices_lock:
        indices = detected_alert_indices.copy()

    language = audio_alert_config.get('language', [])

    return jsonify({
        'indices': indices,
        'objects': [
            audio_alert_config.get('classes', [])[i]
            if i < len(audio_alert_config.get('classes', []))
            else ''
            for i in indices
        ],
        'language': language
    })


# ============================================================
# 获取当前显示类别
# ============================================================

@app.route("/get_display_classes", methods=["GET"])
def get_display_classes_api():
    classes = list(get_display_classes())
    return jsonify({
        "status": "ok",
        "classes": classes
    })


# ============================================================
# SMS 配置接口
# ============================================================

@app.route('/api/sms/config', methods=['POST'])
def set_sms_config():
    if not request.is_json:
        return jsonify({'success': False, 'message': '请求必须是 JSON'}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'message': 'JSON 解析失败'}), 400

    alert_object = data.get('alertObject')
    contact_phone = data.get('contactPhone')

    if not alert_object or not contact_phone:
        return jsonify({'success': False, 'message': '参数不完整'}), 400

    SMS_CONFIG['alert_object'] = alert_object
    SMS_CONFIG['contact_phone'] = contact_phone

    print("✅ 短信配置已更新：", SMS_CONFIG)

    return jsonify({'success': True, 'message': '短信配置已保存'})


@app.route('/api/sms/config/reset', methods=['POST'])
def reset_sms_config():
    try:
        SMS_CONFIG['alert_object'] = None
        SMS_CONFIG['contact_phone'] = None
        print("✅ 短信配置已重置")
        return jsonify({'success': True, 'message': '短信配置已清空'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


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


if __name__ == "__main__":
    start_web()