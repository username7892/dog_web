from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pymysql
import cv2
import numpy as np
import tempfile
import os
import base64

app = Flask(__name__)
CORS(app)

# =========================
# 全局变量
# =========================
current_camera_id = "front camera"  # 默认值
current_filter_classes = []  # 默认空列表，表示不过滤

DB = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "yolo_images",
    "charset": "utf8mb4"
}


# =========================
# 数据库
# =========================

def db():
    return pymysql.connect(**DB)


# =========================
# 图片解码
# =========================

def decode_img(b):
    if not b:
        return None

    if isinstance(b, bytes):
        # 如果数据库保存的是图片路径
        if b.startswith(b'/') or b.startswith(b'images'):
            return cv2.imread(b.decode())
        # 如果数据库保存的是 JPEG 二进制
        return cv2.imdecode(
            np.frombuffer(b, np.uint8),
            cv2.IMREAD_COLOR
        )
    return None


# =========================
# 图片编码
# =========================

def encode(img, t):
    img = img.copy()
    cv2.putText(
        img, t, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 255, 255), 2
    )

    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return b''
    return buf.tobytes()


# =========================
# 接收前端设置的检测类别
# =========================

@app.route('/set_classes', methods=['POST'])
def set_classes():
    global current_filter_classes
    
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"status": "error", "msg": "没有收到JSON数据"}), 400

        classes = data.get("classes", data.get("objects", []))
        if not isinstance(classes, list):
            return jsonify({"status": "error", "msg": "classes 必须是列表"}), 400

        # 清理类别名称
        new_classes = []
        for cls in classes:
            cls = str(cls).strip()
            if cls:
                new_classes.append(cls)

        if len(new_classes) == 0:
            return jsonify({"status": "error", "msg": "至少输入一个有效的检测目标"}), 400

        # 存储到全局变量
        current_filter_classes = new_classes.copy()

        print()
        print("========================================")
        print("回放服务收到检测类别:", current_filter_classes)
        print("========================================")
        print()

        return jsonify({
            "status": "success",
            "msg": "检测类别接收成功",
            "classes": current_filter_classes
        })

    except Exception as e:
        print(f"[ERROR] set_classes: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500


# =========================
# 接收摄像头切换
# =========================

@app.route('/set_camera', methods=['POST'])
def set_camera():
    global current_camera_id
    
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"status": "error", "msg": "没有收到JSON数据"}), 400

        camera_id = data.get("camera_id", "")
        if not camera_id:
            return jsonify({"status": "error", "msg": "缺少camera_id"}), 400

        current_camera_id = camera_id

        print()
        print("========================================")
        print(f"摄像头切换为: {current_camera_id}")
        print("========================================")
        print()

        return jsonify({
            "status": "success",
            "msg": f"摄像头已切换为: {camera_id}",
            "camera_id": camera_id
        })

    except Exception as e:
        print(f"[ERROR] set_camera: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500


# =========================
# 获取检测类别列表
# =========================

@app.route('/api/classes')
def classes():
    conn = db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT DISTINCT class_name
            FROM detections
            ORDER BY class_name
        """)
        result = [row[0] for row in cursor.fetchall()]
        return jsonify(result)
    finally:
        cursor.close()
        conn.close()


# =========================
# 加载历史帧
# =========================

@app.route('/api/load')
def load_frames():
    # 从URL参数获取时间
    start = request.args.get('start')
    end = request.args.get('end')
    
    # 使用全局变量
    camera_id = current_camera_id
    filter_classes = current_filter_classes
    
    if not start or not end:
        return jsonify({"error": "缺少 start 或 end"}), 400
    
    start = start.replace('T', ' ')
    end = end.replace('T', ' ')
    
    conn = db()
    cursor = conn.cursor()
    
    try:
        # 构建SQL查询，同时满足三个条件
        if filter_classes and len(filter_classes) > 0:
            # 三个条件都有：camera_id + 时间范围 + 检测物体
            placeholders = ",".join(["%s"] * len(filter_classes))
            sql = f"""
                SELECT DISTINCT i.id, i.image, i.frame_time
                FROM images i
                JOIN detections d ON i.id = d.image_id
                WHERE i.camera_id = %s
                AND i.frame_time BETWEEN %s AND %s
                AND d.class_name IN ({placeholders})
                ORDER BY i.frame_time
            """
            params = [camera_id, start, end] + filter_classes
        else:
            # 两个条件：camera_id + 时间范围（没有检测物体过滤）
            sql = """
                SELECT id, image, frame_time
                FROM images
                WHERE camera_id = %s
                AND frame_time BETWEEN %s AND %s
                ORDER BY frame_time
            """
            params = [camera_id, start, end]
        
        print(f"[DEBUG] 检索条件:")
        print(f"  camera_id: {camera_id}")
        print(f"  时间范围: {start} ~ {end}")
        print(f"  检测物体: {filter_classes}")
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        # 处理查询结果
        ids = []
        frames = []
        
        for iid, image_data, frame_time in rows:
            img = decode_img(image_data)
            if img is None:
                continue
            jpg = encode(img, frame_time.strftime("%H:%M:%S"))
            if not jpg:
                continue
            ids.append(iid)
            frames.append({
                "id": iid,
                "time": frame_time.strftime("%H:%M:%S"),
                "data": base64.b64encode(jpg).decode("utf-8")
            })
        
        # 查询检测结果
        dets = {}
        if ids:
            placeholders = ",".join(["%s"] * len(ids))
            sql = f"""
                SELECT image_id, class_name, confidence
                FROM detections
                WHERE image_id IN ({placeholders})
                ORDER BY image_id
            """
            cursor.execute(sql, ids)
            for iid, name, conf in cursor.fetchall():
                dets.setdefault(iid, []).append({
                    "name": name,
                    "conf": float(conf)
                })
        
        print(f"[DEBUG] 查询到 {len(frames)} 帧数据")
        
        return jsonify({
            "frames": frames,
            "dets": dets,
            "total": len(frames)
        })
        
    except Exception as e:
        print(f"[ERROR] load_frames: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
        
    finally:
        cursor.close()
        conn.close()


# =========================
# 导出 MP4
# =========================

@app.route('/api/export_mp4')
def export_mp4():
    camera_id = request.args.get('camera_id')
    start = request.args.get('start')
    end = request.args.get('end')

    if not start or not end:
        return "缺少时间参数", 400

    start = start.replace('T', ' ')
    end = end.replace('T', ' ')

    conn = db()
    cursor = conn.cursor()

    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp.close()
    writer = None

    try:
        cursor.execute("""
            SELECT image
            FROM images
            WHERE camera_id = %s
            AND frame_time BETWEEN %s AND %s
            ORDER BY frame_time
        """, (camera_id, start, end))

        for (image_data,) in cursor:
            img = decode_img(image_data)
            if img is None:
                continue
            if writer is None:
                writer = cv2.VideoWriter(
                    tmp.name,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    10,
                    (img.shape[1], img.shape[0])
                )
            writer.write(img)

        if writer is not None:
            writer.release()

        with open(tmp.name, 'rb') as f:
            data = f.read()

        return Response(
            data,
            mimetype='video/mp4',
            headers={'Content-Disposition': 'attachment; filename=playback.mp4'}
        )

    finally:
        cursor.close()
        conn.close()
        if writer is not None:
            writer.release()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# =========================
# 启动
# =========================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)