from web_server import (
    update_frame,
    start_web,
    get_display_classes,
    get_current_classes,
    get_no_alert_classes,
    audio_alert_config,
)
import requests
import queue
import pymysql
import cv2
import os
import time
import threading
import torch
import datetime
import traceback

from ultralytics import YOLOE

camera_id = "front camera"
# ============================================================
# 全局配置
# ============================================================

RTSP_URL = (
    "rtsp://m20-detector:f715e51840a1359d569bbb9a42af402e@120.26.18.138:8554/camera-front"
    #"rtsp://admin:dhlb839.@192.168.50.64:554/Streaming/Channels/101"
)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


# 全局：始终参与检测
SAFETY_CLASSES = {
    "person",
    "helmet",
}

# # 火源类
# FIRE_SOURCES = {
#     "fire",
#     "flame",
#     "spark",
#     "welding",
#     "electric_spark",
# }

# # 易燃物类
# FLAMMABLES = {
#     "gas_tank",
#     "oil_drum",
#     "wood",
#     "paper",
#     "cardboard",
#     "cloth",
#     "flammable_liquid",
# }



# ============================================================
# 图片保存
# ============================================================

SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 摄像头信息
# ============================================================

CAMERA_LOCATIONS = {
    "front_camera": "前摄像头",
}

CAMERA_TYPES = {
    "front_camera": "front camera",
}

# ============================================================
# 数据库队列
# ============================================================

db_queue = queue.Queue(maxsize=200)



def update_detected_indices_to_server(indices):
    """将检测到的索引推送到 web_server"""
    try:
        print(f"[YOLO] 准备推送索引: {indices}")
        response = requests.post(  # 加上 response =
            'http://127.0.0.1:5000/api/update_detected_indices', 
            json={'indices': indices},
            timeout=0.5
        )
        print(f"[YOLO] 推送结果: {response.status_code}")
    except Exception as e:
        print(f"[YOLO] 推送失败: {e}")
# 在检测到告警对象后调用
#update_detected_indices_to_server(current_detected_indices)
# ============================================================
# 摄像头信息
# ============================================================

def get_camera_info(camera_id):
    location = CAMERA_LOCATIONS.get(camera_id, camera_id)
    camera_type = CAMERA_TYPES.get(camera_id, "前摄")
    return location, camera_type







# ============================================================
# 安全帽检测
# ============================================================

def get_yolo_classes(current_classes):
    return list(set(current_classes) | SAFETY_CLASSES)


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
# RTSP RingBuffer
# ============================================================

class RingBuffer:
    def __init__(self, size=50):
        self.size = size
        self.buffer = [None] * size
        self.index = 0
        self.count = 0
        self.lock = threading.Lock()

    def put(self, frame):
        with self.lock:
            self.buffer[self.index] = frame
            self.index = (self.index + 1) % self.size
            self.count = min(self.count + 1, self.size)

    def get(self, delay):
        with self.lock:
            if self.count <= delay:
                return None
            idx = (self.index - delay - 1) % self.size
            return self.buffer[idx]

# ============================================================
# RTSP
# ============================================================

class DualRTSP:
    def __init__(self, url, buffer):
        self.url = url
        self.buffer = buffer
        self.cap1 = None
        self.cap2 = None
        self.active = 1
        self.running = True
        self.fixing = None

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None

    def _fix_cap(self, num):
        while self.running and self.fixing == num:
            cap = self._open()
            if cap:
                if num == 1:
                    self.cap1 = cap
                else:
                    self.cap2 = cap
                self.fixing = None
                return
            time.sleep(1)

    def run(self):
        self.cap1 = self._open()
        time.sleep(0.3)
        self.cap2 = self._open()

        if not self.cap1 and not self.cap2:
            self.running = False
            print("[RTSP] 两个连接都失败")
            return

        while self.running:
            cap = self.cap1 if self.active == 1 else self.cap2
            if cap is None:
                self._switch()
                continue

            ret, frame = cap.read()
            if ret and frame is not None:
                self.buffer.put(frame)
            else:
                print(f"[RTSP] 摄像头 {self.active} 读取失败，重新连接")
                if self.active == 1:
                    if self.cap1:
                        self.cap1.release()
                    self.cap1 = None
                    if self.fixing != 1:
                        self.fixing = 1
                        threading.Thread(target=self._fix_cap, args=(1,), daemon=True).start()
                else:
                    if self.cap2:
                        self.cap2.release()
                    self.cap2 = None
                    if self.fixing != 2:
                        self.fixing = 2
                        threading.Thread(target=self._fix_cap, args=(2,), daemon=True).start()
                self._switch()

    def _switch(self):
        self.active = 2 if self.active == 1 else 1

    def stop(self):
        self.running = False
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()

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
            # 先查询该摄像头当前最大的 camera_seq
            cursor.execute(
                """
                SELECT COALESCE(MAX(camera_seq), 0) + 1
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

        except Exception as e:
            print("[DB ERROR]", e)
            db.rollback()
        finally:
            db_queue.task_done()

# ============================================================
# 主程序
# ============================================================

def main():
    camera_id = "front_camera"
    location, camera_type = get_camera_info(camera_id)

    # GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*32}\n设备: {device}\n{'='*32}\n")

    # YOLOE
    print("[YOLO] 正在加载 YOLOE...")
    model = YOLOE("yoloe-v8l-seg.pt")
    model.to(device)
    print("[YOLO] YOLOE 加载完成")

    # 初始化检测类别
    current_classes = ["person", "car", "dog", "cup", "phone"]
    current_classes = list(dict.fromkeys(current_classes))
    print(f"\n{'='*32}\n【INIT】初始检测类别:\n{current_classes}\n{'='*32}\n")

    try:
        yolo_classes = get_yolo_classes(current_classes)
        model.set_classes(yolo_classes)
        print("[YOLO] 初始类别设置成功:", yolo_classes)
    except Exception:
        print("[YOLO] 初始 set_classes 失败:")
        traceback.print_exc()

    # RTSP
    buffer = RingBuffer(50)
    rtsp = DualRTSP(RTSP_URL, buffer)
    threading.Thread(target=rtsp.run, daemon=True).start()

    # Web
    threading.Thread(target=start_web, daemon=True).start()

    # 数据库线程
    threading.Thread(target=database_worker, daemon=True).start()

    time.sleep(3)

    # 参数
    delay = 15
    last_save = 0
    last_abnormal = 0
    save_interval = 2
    abnormal_interval = 5
    class_update_interval = 0.5
    last_class_check = 0

    detected_alert_indices = []
    detected_indices_lock = threading.Lock()
    # =========================
    # ✅ 安全帽持续未佩戴计时器
    # =========================
    no_helmet_timer = {}  # person_id -> start_time
    NO_HELMET_ALERT_SECONDS = 3.0
    # 主循环
    while True:
        frame = buffer.get(delay)
        if frame is None:
            time.sleep(0.01)
            continue

        try:
            now = time.time()

            # =========================
            # ① 获取告警配置
            # =========================
            try:
                alert_config_response = requests.get(
                    'http://127.0.0.1:5000/api/get_audio_alert_config',
                    timeout=0.5
                )
                if alert_config_response.status_code == 200:
                    alert_config = alert_config_response.json()
                else:
                    alert_config = {'classes': []}
            except Exception:
                alert_config = {'classes': []}

            alert_classes = alert_config.get('classes', [])

            # =========================
            # ② person / person without helmet → 强制注入检测类
            # =========================
            need_person_alert = "person" in alert_classes
            need_helmet_check = "person without helmet" in alert_classes

            # 注意：只往 current_classes 加 YOLOE 真能检的实体类
            if need_person_alert and "person" not in current_classes:
                current_classes = list(dict.fromkeys(current_classes + ["person"]))

            if need_helmet_check:
                if "person" not in current_classes:
                    current_classes = list(dict.fromkeys(current_classes + ["person"]))
                if "helmet" not in current_classes:
                    current_classes = list(dict.fromkeys(current_classes + ["helmet"]))
                    print("[YOLO] 注入 helmet（因 person without helmet 告警）")

            # =========================
            # ③ 定时更新检测类别
            # =========================
            if now - last_class_check >= class_update_interval:
                last_class_check = now
                try:
                    new_classes = get_current_classes()
                except Exception as e:
                    print("[CLASS ERROR]", e)
                    new_classes = current_classes

                if new_classes is None:
                    new_classes = []

                new_classes = [str(c) for c in new_classes if str(c).strip()]
                new_classes = list(dict.fromkeys(new_classes))

                if new_classes != current_classes:
                    print(f"\n{'='*32}\n【YOLO CLASS UPDATE】\n旧类别: {current_classes}\n新类别: {new_classes}\n{'='*32}\n")
                    current_classes = new_classes.copy()
                    if current_classes:
                        try:
                            yolo_classes = get_yolo_classes(current_classes)
                            model.set_classes(yolo_classes)
                            print("[YOLO] set_classes 成功:", yolo_classes)
                        except Exception as e:
                            print("[YOLO] set_classes 失败:", e)
                            traceback.print_exc()

            # =========================
            # ④ 不报警对象
            # =========================
            try:
                no_alert_classes = get_no_alert_classes()
            except Exception as e:
                print("[NO ALERT ERROR]", e)
                no_alert_classes = set()
            if no_alert_classes is None:
                no_alert_classes = set()
            no_alert_classes = set(no_alert_classes)

            # =========================
            # ⑤ YOLOE 推理
            # =========================
            res = model.predict(frame, conf=0.6, verbose=False)[0]

            all_objects = []
            for box in res.boxes:
                cls_id = int(box.cls[0])
                name = res.names.get(cls_id, str(cls_id))

                if name not in current_classes:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                all_objects.append({
                    "name": name,
                    "conf": conf,
                    "box": [x1, y1, x2, y2]
                })

            print(f"[DEBUG] all_objects names: {[o['name'] for o in all_objects]}")
            print(f"[DEBUG] current_classes: {current_classes}")
            print(f"[DEBUG] alert_classes: {alert_classes}")

            # =========================
            # ⑥ 告警索引（核心）
            # =========================
            current_detected_indices = []

            # ✅ 6.1 "person" 在告警列表 → 检测到人就立即告警
            if need_person_alert:
                if any(o["name"] == "person" for o in all_objects):
                    idx = alert_classes.index("person")
                    if idx not in current_detected_indices:
                        current_detected_indices.append(idx)

            # ✅ 6.2 "person without helmet" → 持续3秒未戴头盔才告警
            abnormal = False
            abnormal_conf = 0.0
            person_obj = None

            if need_helmet_check:
                single_frame_abnormal, person_obj = check_no_hat(all_objects)

                if person_obj:
                    x1, y1, x2, y2 = person_obj["box"]
                    # 头部中心 + 量化抗抖动（10px 容差）
                    head_cx = int((x1 + x2) / 2)
                    head_cy = int(y1 + (y2 - y1) * 0.2)
                    person_id = (head_cx // 10, head_cy // 10)

                    if single_frame_abnormal:
                        if person_id not in no_helmet_timer:
                            no_helmet_timer[person_id] = now
                        duration = now - no_helmet_timer[person_id]
                        if duration >= NO_HELMET_ALERT_SECONDS:
                            abnormal = True
                            abnormal_conf = float(person_obj["conf"])
                    else:
                        # 只要有一帧判定为"戴了"，立即清零
                        no_helmet_timer.pop(person_id, None)

                # 清理离开画面的人
                no_helmet_timer = {
                    pid: t for pid, t in no_helmet_timer.items()
                    if now - t < 10
                }

                # ✅ 未戴安全帽 → 注入索引（不依赖 obj_name 匹配）
                if abnormal:
                    idx = alert_classes.index("person without helmet")
                    if idx not in current_detected_indices:
                        current_detected_indices.append(idx)

            # =========================
            # ⑦ 推送索引
            # =========================
            update_detected_indices_to_server(current_detected_indices)

            if current_detected_indices:
                print(f"[ALERT] 检测到告警对象索引: {current_detected_indices}")
                print(f"[ALERT] 对应对象: {[alert_classes[i] for i in current_detected_indices]}")

            # =========================
            # ⑧ 可视化
            # =========================
            vis = frame.copy()
            for obj in all_objects:
                x1, y1, x2, y2 = obj["box"]
                name = obj["name"]

                if name == "helmet":
                    color = (0, 165, 255)
                elif name in no_alert_classes:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)

                if abnormal and name == "person" and obj is person_obj:
                    color = (0, 0, 255)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {obj['conf']:.2f}"
                if name in no_alert_classes:
                    label += " [NO ALERT]"
                cv2.putText(vis, label, (x1, max(y1 - 6, 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(vis, "Detecting: " + ", ".join(current_classes[:5]),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if abnormal:
                cv2.putText(vis, "WARNING: No Helmet!", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # =========================
            # ⑨ 保存 & 入库
            # =========================
            save = False
            save_abnormal = False

            if all_objects:
                if abnormal:
                    if now - last_abnormal > abnormal_interval:
                        save_abnormal = True
                        last_abnormal = now
                else:
                    if now - last_save > save_interval:
                        save = True
                        last_save = now

            if save or save_abnormal:
                frame_time = datetime.datetime.now()
                prefix = "abnormal_" if save_abnormal else "normal_"
                filename = f"{prefix}{camera_id}_{int(frame_time.timestamp() * 1000)}.jpg"
                path = os.path.join(SAVE_DIR, filename)
                image_saved = cv2.imwrite(path, vis)

                if image_saved:
                    db_queue.put({
                        "camera_id": camera_id,
                        "image_path": path,
                        "frame_time": frame_time,
                        "objects": all_objects,
                        "abnormal": abnormal,
                        "abnormal_conf": abnormal_conf,
                        "location": location,
                        "camera_type": camera_type,
                    })
                    if save_abnormal:
                        print("[ALERT] 检测到未佩戴安全帽！")
                        print("图片:", path)
                else:
                    print("[IMAGE SAVE ERROR]", path)

            update_frame(vis)

        except Exception as e:
            print("[MAIN ERROR]", e)
            traceback.print_exc()

    rtsp.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()