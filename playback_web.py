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
        img,
        t,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    ok, buf = cv2.imencode(
        '.jpg',
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 80]
    )

    if not ok:
        return b''

    return buf.tobytes()


# =========================
# 获取检测类别
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

        result = [
            row[0]
            for row in cursor.fetchall()
        ]

        return jsonify(result)

    finally:

        cursor.close()
        conn.close()


# =========================
# 加载历史帧
# =========================

@app.route('/api/load')
def load_frames():

    camera_id = request.args.get(
        'camera_id',
        'cam01'
    )

    start = request.args.get('start')
    end = request.args.get('end')

    if not start or not end:
        return jsonify({
            "error": "缺少 start 或 end"
        }), 400

    start = start.replace('T', ' ')
    end = end.replace('T', ' ')

    conn = db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            SELECT
                id,
                image,
                frame_time
            FROM images
            WHERE camera_id = %s
            AND frame_time BETWEEN %s AND %s
            ORDER BY frame_time
        """, (
            camera_id,
            start,
            end
        ))

        rows = cursor.fetchall()

        ids = []
        frames = []

        for iid, image_data, frame_time in rows:

            img = decode_img(image_data)

            if img is None:
                continue

            jpg = encode(
                img,
                frame_time.strftime("%H:%M:%S")
            )

            if not jpg:
                continue

            ids.append(iid)

            frames.append({
                "id": iid,
                "time": frame_time.strftime(
                    "%H:%M:%S"
                ),
                "data": base64.b64encode(
                    jpg
                ).decode("utf-8")
            })

        # =========================
        # 查询检测结果
        # =========================

        dets = {}

        if ids:

            placeholders = ",".join(
                ["%s"] * len(ids)
            )

            sql = f"""
                SELECT
                    image_id,
                    class_name,
                    confidence
                FROM detections
                WHERE image_id IN ({placeholders})
                ORDER BY image_id
            """

            cursor.execute(
                sql,
                ids
            )

            for iid, name, conf in cursor.fetchall():

                dets.setdefault(
                    iid,
                    []
                ).append({
                    "name": name,
                    "conf": float(conf)
                })

        return jsonify({
            "frames": frames,
            "dets": dets
        })

    finally:

        cursor.close()
        conn.close()


# =========================
# 导出 MP4
# =========================

@app.route('/api/export_mp4')
def export_mp4():

    camera_id = request.args.get(
        'camera_id',
        'cam01'
    )

    start = request.args.get('start')
    end = request.args.get('end')

    if not start or not end:
        return "缺少时间参数", 400

    start = start.replace('T', ' ')
    end = end.replace('T', ' ')

    conn = db()
    cursor = conn.cursor()

    tmp = tempfile.NamedTemporaryFile(
        suffix='.mp4',
        delete=False
    )

    tmp.close()

    writer = None

    try:

        cursor.execute("""
            SELECT image
            FROM images
            WHERE camera_id = %s
            AND frame_time BETWEEN %s AND %s
            ORDER BY frame_time
        """, (
            camera_id,
            start,
            end
        ))

        for (image_data,) in cursor:

            img = decode_img(image_data)

            if img is None:
                continue

            if writer is None:

                writer = cv2.VideoWriter(
                    tmp.name,
                    cv2.VideoWriter_fourcc(
                        *'mp4v'
                    ),
                    10,
                    (
                        img.shape[1],
                        img.shape[0]
                    )
                )

            writer.write(img)

        if writer is not None:
            writer.release()

        with open(
            tmp.name,
            'rb'
        ) as f:
            data = f.read()

        return Response(
            data,
            mimetype='video/mp4',
            headers={
                'Content-Disposition':
                'attachment; filename=playback.mp4'
            }
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

    app.run(
        host='0.0.0.0',
        port=5002,
        debug=False
    )