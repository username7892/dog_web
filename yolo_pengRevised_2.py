from web_server_2 import update_frame, start_web, get_display_classes

import queue
import pymysql
import cv2
import os
import time
import threading
import torch
import datetime
import traceback
import requests

from ultralytics import YOLOE

camera_id = "rear camera"

#安全帽判断

def check_no_hat(objects):
    persons = [o for o in objects if o["name"] == "person"]
    helmets = [o for o in objects if o["name"] == "helmet"]

    if not persons:
        return False, None

    # 给每个人计算头部区域
    person_heads = []
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        h_y1 = y1
        h_y2 = y1 + (y2 - y1) * 0.3
        person_heads.append({
            "person": p,
            "x1": x1, "x2": x2,
            "y1": h_y1, "y2": h_y2
        })

    # 每个安全帽匹配最近的人
    matched_person_indices = set()

    for h in helmets:
        hx1, hy1, hx2, hy2 = h["box"]
        cx = (hx1 + hx2) / 2
        cy = (hy1 + hy2) / 2

        best_p = None
        best_dist = float("inf")

        for i, head in enumerate(person_heads):
            if head["x1"] < cx < head["x2"] and head["y1"] < cy < head["y2"]:
                dist = abs(cy - (head["y1"] + head["y2"]) / 2)
                if dist < best_dist:
                    best_dist = dist
                    best_p = i

        if best_p is not None:
            matched_person_indices.add(best_p)

    # 找第一个没匹配到帽子的人
    for i, head in enumerate(person_heads):
        if i not in matched_person_indices:
            return True, head["person"]

    return False, None

# ============================================================
# 全局配置
# ============================================================

RTSP_URL = (
    "rtsp://m20-detector:f715e51840a1359d569bbb9a42af402e@120.26.18.138:8554/camera-rear"
    #"rtsp://admin:dhlb839.@192.168.50.64:554/Streaming/Channels/102"
)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# 图片保存目录
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 摄像头配置
CAMERA_LOCATIONS = {
    "rear_camera": "后摄像头",
}

CAMERA_TYPES = {
    "rear_camera": "rear camera"
}

# 数据库队列
db_queue = queue.Queue(maxsize=200)


# ============================================================
# 摄像头信息
# ============================================================

def get_camera_info(camera_id):
    location = CAMERA_LOCATIONS.get(camera_id, camera_id)
    camera_type = CAMERA_TYPES.get(camera_id, "前摄")
    return location, camera_type


# ============================================================
# 简易帧缓冲
# ============================================================

class FrameBuffer:
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def put(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def get(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


# ============================================================
# 双 RTSP 连接
# ============================================================

class DualRTSP:

    def __init__(self, url, buffer: FrameBuffer):
        self.url = url
        self.buffer = buffer
        self.cap1 = None
        self.cap2 = None
        self.active = 1
        self.running = True
        self.fixing = None
        self.lock = threading.Lock()

    def _open(self):
        print("[RTSP] 正在连接:", self.url)
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            print("[RTSP] 连接成功")
            return cap
        print("[RTSP] 连接失败")
        cap.release()
        return None

    def _fix_cap(self, num):
        while self.running and self.fixing == num:
            cap = self._open()
            if cap:
                with self.lock:
                    if num == 1:
                        self.cap1 = cap
                    else:
                        self.cap2 = cap
                    self.fixing = None
                print(f"[RTSP] cap{num} 重连成功")
                return
            time.sleep(1)

    def run(self):
        self.cap1 = self._open()
        time.sleep(0.3)
        self.cap2 = self._open()

        if self.cap1 is None and self.cap2 is None:
            print("[RTSP] 两路连接全部失败")
            self.running = False
            return

        while self.running:
            cap = self.cap1 if self.active == 1 else self.cap2

            if cap is None:
                self._switch()
                time.sleep(0.05)
                continue

            ret, frame = cap.read()

            if ret and frame is not None:
                self.buffer.put(frame)
            else:
                print(f"[RTSP] cap{self.active} 读取失败，准备切换")

                if self.active == 1:
                    try:
                        self.cap1.release()
                    except Exception:
                        pass
                    self.cap1 = None
                    if self.fixing is None:
                        self.fixing = 1
                        threading.Thread(target=self._fix_cap, args=(1,), daemon=True).start()
                else:
                    try:
                        self.cap2.release()
                    except Exception:
                        pass
                    self.cap2 = None
                    if self.fixing is None:
                        self.fixing = 2
                        threading.Thread(target=self._fix_cap, args=(2,), daemon=True).start()

                self._switch()
                time.sleep(0.05)

    def _switch(self):
        self.active = 2 if self.active == 1 else 1
        print(f"[RTSP] 切换到 cap{self.active}")

    def stop(self):
        self.running = False
        for c in (self.cap1, self.cap2):
            try:
                if c:
                    c.release()
            except Exception:
                pass

# ============================================================
# 数据库线程
# ============================================================

def database_worker():
    print("[DB] 数据库线程启动")
    db = pymysql.connect(
        host="localhost",
        user="root",
        password="123456",
        database="yolo_images",
        charset="utf8mb4"
    )
    cursor = db.cursor()

    while True:
        data = db_queue.get()
        try:
            camera_id = data["camera_id"]

            # 查询该摄像头当前最大的 camera_seq，没有则从0开始
            cursor.execute(
                """
                SELECT COALESCE(MAX(camera_seq), -1) + 1
                FROM images
                WHERE camera_id = %s
                """,
                (camera_id,)
            )
            next_seq = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO images
                (camera_id, camera_seq, image, frame_time)
                VALUES (%s,%s,%s,%s)
                """,
                (
                    camera_id,
                    next_seq,
                    data["image_path"],
                    data["frame_time"]
                )
            )

            image_id = cursor.lastrowid

            for obj in data["objects"]:
                x1, y1, x2, y2 = obj["box"]
                cursor.execute(
                    """
                    INSERT INTO detections
                    (image_id, class_name, confidence,
                     x1, y1, x2, y2)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        image_id,
                        obj["name"],
                        obj["conf"],
                        x1, y1, x2, y2
                    )
                )

            if data["abnormal"]:
                cursor.execute(
                    """
                    INSERT INTO abnormal_events
                    (image_id, event_type, description,
                     camera_id, confidence)
                    VALUES (%s,%s,%s,%s,%s)
                    """,
                    (
                        image_id,
                        "no_hat",
                        "人员未佩戴安全帽",
                        camera_id,
                        data["abnormal_conf"]
                    )
                )

            db.commit()
            print(f"[DB] 保存成功 image_id={image_id}, camera_id={camera_id}, camera_seq={next_seq}")

        except Exception as e:
            print("[DB ERROR]", e)
            db.rollback()
        finally:
            db_queue.task_done()


# ============================================================
# 安全入队
# ============================================================

def put_db_queue(data):
    try:
        db_queue.put_nowait(data)
    except queue.Full:
        print("[DB] 队列已满，本次数据库记录跳过")


# ============================================================
# 绘制检测框
# ============================================================

def draw_detections(frame, objects, abnormal):
    vis = frame.copy()

    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        name = obj["name"]
        conf = obj["conf"]

        if name == "person" and abnormal:
            color = (0, 0, 255)
        elif name == "helmet":
            color = (255, 0, 0)
        elif name == "person":
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.2f}"
        text_y = max(y1 - 8, 20)
        cv2.putText(vis, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    if abnormal:
        cv2.putText(vis, "WARNING: NO HELMET", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)

    return vis


# ============================================================
# 主程序
# ============================================================

def main():
    camera_id = "rear_camera"
    location, camera_type = get_camera_info(camera_id)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 40)
    print(f"[SYSTEM] device: {device}")
    print(f"[SYSTEM] camera: {camera_id}")
    print(f"[SYSTEM] location: {location}")
    print(f"[SYSTEM] camera_type: {camera_type}")
    print("=" * 40)

    print("[YOLO] 正在加载模型...")
    model = YOLOE("yoloe-v8l-seg.pt")
    model.to(device)
    print("[YOLO] 模型加载完成")

    buffer = FrameBuffer()

    rtsp = DualRTSP(RTSP_URL, buffer)
    threading.Thread(target=rtsp.run, daemon=True).start()

    threading.Thread(target=start_web, daemon=True).start()

    threading.Thread(target=database_worker, daemon=True).start()

    time.sleep(3)

    last_save = 0
    last_abnormal = 0
    save_interval = 2
    abnormal_interval = 10
    last_classes = None

    target_fps = 15
    frame_interval = 1.0 / target_fps
    last_infer_time = 0
    last_detected_classes = None
    last_object_save_time = 0
    last_abnormal_save_time = 0

    last_objects = []
    last_abnormal = False
    last_vis_frame = None

    print("[MAIN] 检测线程启动")

    # =========================
    # ✅ 安全帽时间计数器
    # =========================
    no_helmet_start_time = {}  # person_id -> timestamp
    NO_HELMET_ALERT_SECONDS = 3.0  # 持续多少秒才报警
    while True:
        try:
            frame = buffer.get()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()

            # =========================
            # ① 显示类别
            # =========================
            display_classes = get_display_classes()
            if display_classes is None:
                display_classes = []
            display_classes = list(display_classes)

            # =========================
            # ② 告警配置
            # =========================
            try:
                alert_config_resp = requests.get(
                    'http://127.0.0.1:5000/api/get_audio_alert_config',
                    timeout=0.5
                )
                if alert_config_resp.status_code == 200:
                    alert_config = alert_config_resp.json()
                else:
                    alert_config = {'classes': []}
            except:
                alert_config = {'classes': []}

            alert_classes = alert_config.get('classes', [])

            # =========================
            # ③ 强制检测 person / helmet
            # =========================
            need_person = "person" in alert_classes
            need_helmet_check = "person without helmet" in alert_classes

            if need_helmet_check or need_person:
                if "person" not in display_classes:
                    display_classes.append("person")
                if need_helmet_check and "helmet" not in display_classes:
                    display_classes.append("helmet")

            # =========================
            # ④ set_classes（原逻辑）
            # =========================
            if display_classes != last_classes:
                print("\n" + "=" * 40)
                print(f"[WEB] 前端选择类别: {display_classes}")
                print("=" * 40)

                if display_classes:
                    try:
                        model.set_classes(display_classes)
                        last_classes = display_classes.copy()
                        print("[YOLO] set_classes 成功")
                    except Exception as exc:
                        print("[YOLO] set_classes 失败:", exc)
                        traceback.print_exc()
                        update_frame(frame)
                        continue
                else:
                    last_classes = []

            if not display_classes:
                update_frame(frame)
                time.sleep(0.001)
                continue

            if now - last_infer_time < frame_interval:
                if last_vis_frame is not None:
                    update_frame(last_vis_frame)
                else:
                    update_frame(frame)
                continue

            last_infer_time = now

            # =========================
            # ⑤ YOLOE 推理
            # =========================
            results = model.predict(frame, conf=0.6, verbose=False)
            if not results:
                update_frame(frame)
                continue

            res = results[0]
            objects = []

            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id < 0 or cls_id >= len(display_classes):
                    continue

                name = display_classes[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                objects.append({
                    "name": name,
                    "conf": conf,
                    "box": [x1, y1, x2, y2]
                })

            # =========================
            # ⑥ 告警索引（核心）
            # =========================
            current_detected_indices = []

            # ✅ 6.1 person 直接告警
            if "person" in alert_classes:
                if any(o["name"] == "person" for o in objects):
                    idx = alert_classes.index("person")
                    if idx not in current_detected_indices:
                        current_detected_indices.append(idx)

            # ✅ 6.2 person without helmet（稳定 3 秒）
            abnormal = False
            abnormal_conf = 0.0
            person_obj = None

            if need_helmet_check:
                single_frame_abnormal, person_obj = check_no_hat(objects)

                if person_obj:
                    # ✅ 头部中心 ID（抗抖动）
                    x1, y1, x2, y2 = person_obj["box"]
                    head_cx = int((x1 + x2) / 2)
                    head_cy = int(y1 + (y2 - y1) * 0.2)
                    person_id = (head_cx // 10, head_cy // 10)

                    if single_frame_abnormal:
                        if person_id not in no_helmet_start_time:
                            no_helmet_start_time[person_id] = now

                        if now - no_helmet_start_time[person_id] >= NO_HELMET_ALERT_SECONDS:
                            abnormal = True
                            abnormal_conf = float(person_obj["conf"])
                    else:
                        # ✅ 只要一帧戴了，立刻清零
                        no_helmet_start_time.pop(person_id, None)

                # ✅ 清理离开画面的人
                no_helmet_start_time = {
                    pid: t for pid, t in no_helmet_start_time.items()
                    if now - t < 10
                }

                # ✅ 未戴安全帽 → 索引
                if abnormal:
                    idx = alert_classes.index("person without helmet")
                    if idx not in current_detected_indices:
                        current_detected_indices.append(idx)

            # =========================
            # ⑦ 推送索引
            # =========================
            try:
                requests.post(
                    'http://127.0.0.1:5000/api/update_detected_indices',
                    json={'indices': current_detected_indices},
                    timeout=0.5
                )
            except:
                pass

            if current_detected_indices:
                print(f"[ALERT] 检测到告警对象索引: {current_detected_indices}")

            # =========================
            # ⑧ 可视化 / 保存 / 入库（原逻辑）
            # =========================
            last_objects = objects
            last_abnormal_state = abnormal

            vis_frame = draw_detections(frame, objects, abnormal)
            last_vis_frame = vis_frame

            should_save = False
            save_abnormal = False

            current_classes = set(obj["name"] for obj in objects)

            if current_classes:
                if last_detected_classes is None:
                    should_save = True
                    last_detected_classes = set()
                    last_object_save_time = 0
                else:
                    if now - last_object_save_time >= 1.0:
                        should_save = True
                        last_object_save_time = now

                last_detected_classes = last_detected_classes | current_classes

            if abnormal:
                if now - last_abnormal_save_time >= 1.0:
                    save_abnormal = True
                    last_abnormal_save_time = now

            if should_save or save_abnormal:
                frame_time = datetime.datetime.now()
                filename = f"{camera_id}_{int(frame_time.timestamp() * 1000)}.jpg"
                path = os.path.join(SAVE_DIR, filename)

                if cv2.imwrite(path, vis_frame):
                    print("[IMAGE] 保存成功:", path)

                    put_db_queue({
                        "camera_id": camera_id,
                        "image_path": path,
                        "frame_time": frame_time,
                        "objects": objects,
                        "abnormal": abnormal,
                        "abnormal_conf": abnormal_conf,
                        "location": location,
                        "camera_type": camera_type,
                    })
                else:
                    print("[IMAGE SAVE ERROR]", path)

            update_frame(vis_frame)

            try:
                cv2.imshow("rear RTSP", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                pass

        except Exception as exc:
            print("\n[MAIN ERROR]", exc)
            traceback.print_exc()
            try:
                update_frame(frame)
            except Exception:
                pass
            time.sleep(0.01)

    print("[MAIN] 正在退出...")
    rtsp.stop()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
if __name__ == "__main__":
    main()