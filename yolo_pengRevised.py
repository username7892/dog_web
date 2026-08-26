from web_server import (
    update_frame,
    start_web,
    get_display_classes,
    get_current_classes,
    get_no_alert_classes,
    audio_alert_config,
    SMS_CONFIG
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
from alert_center import AlertCenter,send_sms


# ============================================================
# 全局配置
# ============================================================
camera_id = "front_camera"
RTSP_URL = "rtsp://admin:dhlb839.@192.168.50.64:554/Streaming/Channels/101"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
SAFETY_CLASSES = {""}
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)
CAMERA_LOCATIONS = {"front_camera": "前摄像头"}
CAMERA_TYPES = {"front_camera": "front camera"}
db_queue = queue.Queue(maxsize=200)
alert_center = AlertCenter(cooldown_seconds=300)
# ============================================================
# 基础工具
# ============================================================
def get_camera_info(camera_id):
    return CAMERA_LOCATIONS.get(camera_id, camera_id), CAMERA_TYPES.get(camera_id, "前摄")

def get_yolo_classes(current_classes):
    return list(set(current_classes) | SAFETY_CLASSES)

# ============================================================
# 安全帽判断
# ============================================================
def check_no_hat(objects):
    persons = [o for o in objects if o["name"] == "person"]
    helmets = [o for o in objects if o["name"] == "helmet"]
    if not persons:
        return False, None
    person_heads = []
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        person_heads.append({
            "person": p,
            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y1 + (y2 - y1) * 0.3
        })
    matched_person_indices = set()
    for h in helmets:
        hx1, hy1, hx2, hy2 = h["box"]
        cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        best_p, best_dist = None, float("inf")
        for i, head in enumerate(person_heads):
            if head["x1"] < cx < head["x2"] and head["y1"] < cy < head["y2"]:
                dist = abs(cy - (head["y1"] + head["y2"]) / 2)
                if dist < best_dist:
                    best_dist, best_p = dist, i
        if best_p is not None:
            matched_person_indices.add(best_p)
    for i, head in enumerate(person_heads):
        if i not in matched_person_indices:
            return True, head["person"]
    return False, None

# ============================================================
# RTSP RingBuffer
# ============================================================
class RingBuffer:
    def __init__(self, size=50):
        self.size, self.buffer = size, [None] * size
        self.index = self.count = 0
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
            return self.buffer[(self.index - delay - 1) % self.size]

# ============================================================
# Dual RTSP
# ============================================================
class DualRTSP:
    def __init__(self, url, buffer):
        self.url, self.buffer = url, buffer
        self.cap1 = self.cap2 = None
        self.active = 1
        self.running = True
        self.fixing = None

    def _open(self):
        print("[RTSP] 正在连接:", self.url)
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            print("[RTSP] 连接成功")
            return cap
        return None

    def _fix_cap(self, num):
        while self.running and self.fixing == num:
            cap = self._open()
            if cap:
                setattr(self, f"cap{num}", cap)
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
            if not cap:
                self._switch()
                time.sleep(0.05)
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
# 告警索引推送
# ============================================================
def update_detected_indices_to_server(indices):
    try:
        print(f"[YOLO] 准备推送索引: {indices}")
        r = requests.post("http://127.0.0.1:5000/api/update_detected_indices",
                          json={"indices": indices}, timeout=0.5)
        print(f"[YOLO] 推送结果: {r.status_code}")
    except Exception as e:
        print(f"[YOLO] 推送失败: {e}")

# ============================================================
# 数据库线程
# ============================================================
def database_worker():
    print("[DB] 数据库线程启动")
    db = pymysql.connect(host="localhost", user="root", password="123456",
                         database="yolo_images", charset="utf8mb4")
    cursor = db.cursor()
    while True:
        data = db_queue.get()
        try:
            cid = data["camera_id"]
            cursor.execute("SELECT COALESCE(MAX(camera_seq),0)+1 FROM images WHERE camera_id=%s", (cid,))
            seq = cursor.fetchone()[0]
            cursor.execute("INSERT INTO images (camera_id,camera_seq,image,frame_time) VALUES (%s,%s,%s,%s)",
                           (cid, seq, data["image_path"], data["frame_time"]))
            image_id = cursor.lastrowid
            for obj in data["objects"]:
                x1, y1, x2, y2 = obj["box"]
                cursor.execute(
                    "INSERT INTO detections (image_id,class_name,confidence,x1,y1,x2,y2) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (image_id, obj["name"], obj["conf"], x1, y1, x2, y2))
            if data["abnormal"]:
                cursor.execute(
                    "INSERT INTO abnormal_events (image_id,event_type,description,camera_id,confidence) "
                    "VALUES (%s,%s,%s,%s,%s)",
                    (image_id, "no_hat", "人员未佩戴安全帽", cid, data["abnormal_conf"]))
            db.commit()
            print(f"[DB] 保存成功 image_id={image_id}, camera_id={cid}, camera_seq={seq}")
        except Exception as e:
            print("[DB ERROR]", e)
            db.rollback()
        finally:
            db_queue.task_done()

# ============================================================
# 主程序
# ============================================================
def main():
    current_camera_id = "front_camera"
    location, camera_type = get_camera_info(current_camera_id)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 40)
    print(f"[SYSTEM] device: {device}")
    print(f"[SYSTEM] camera: {current_camera_id}")
    print(f"[SYSTEM] location: {location}")
    print(f"[SYSTEM] camera_type: {camera_type}")
    print("=" * 40)

    print("[YOLO] 正在加载 YOLOE...")
    model = YOLOE("yoloe-v8l-seg.pt").to(device)
    print("[YOLO] YOLOE 加载完成")

    current_classes = list(dict.fromkeys(["person", "car", "dog", "cup", "phone"]))
    initial_classes = get_yolo_classes(current_classes)
    print("\n" + "=" * 40)
    print("[INIT] 初始检测类别:")
    print(initial_classes)
    print("=" * 40)

    try:
        model.set_classes(initial_classes)
        current_classes = initial_classes.copy()
        print("[YOLO] 初始 set_classes 成功")
    except Exception:
        print("[YOLO] 初始 set_classes 失败")
        traceback.print_exc()

    buffer = RingBuffer(50)
    rtsp = DualRTSP(RTSP_URL, buffer)
    threading.Thread(target=rtsp.run, daemon=True).start()
    threading.Thread(target=start_web, daemon=True).start()
    threading.Thread(target=database_worker, daemon=True).start()
    time.sleep(3)

    delay = 15
    last_save = last_abnormal = 0
    save_interval = 2
    abnormal_interval = 5
    class_update_interval = 0.5
    last_class_check = 0
    no_helmet_timer = {}
    NO_HELMET_ALERT_SECONDS = 2.0

    while True:
        print("AK_ID:", os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"))
        print("AK_SECRET:", "已加载" if os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET") else "未加载")

        frame = buffer.get(delay)
        if frame is None:
            time.sleep(0.01)
            continue
        try:
            now = time.time()

            try:
                r = requests.get("http://127.0.0.1:5000/api/get_audio_alert_config", timeout=0.5)
                alert_config = r.json() if r.status_code == 200 else {"classes": []}
            except Exception:
                alert_config = {"classes": []}
            alert_classes = alert_config.get("classes", [])

            need_person_alert = "person" in alert_classes
            need_helmet_check = "person without helmet" in alert_classes

            if now - last_class_check >= class_update_interval:
                last_class_check = now

                try:
                    frontend_classes = get_current_classes() or []
                except Exception as e:
                    print("[CLASS ERROR]", e)
                    frontend_classes = []

                frontend_classes = [str(c).strip() for c in frontend_classes if str(c).strip()]
                frontend_classes = list(dict.fromkeys(frontend_classes))

                alert_str_2 = SMS_CONFIG.get("alert_object", "")
                alert_classes_2 = {x.strip() for x in alert_str_2.split(",") if x.strip()} if alert_str_2 else set()

                final_classes = set(frontend_classes) | SAFETY_CLASSES
                if need_person_alert:
                    final_classes.add("person")
                if need_helmet_check:
                    final_classes.update(["person", "helmet"])
                final_classes |= alert_classes_2
                final_classes = sorted(final_classes)

                if set(final_classes) != set(current_classes):
                    print("\n" + "=" * 60)
                    print("[YOLO CLASS UPDATE]")
                    print("前端类别:", frontend_classes)
                    print("音频告警:", alert_classes)
                    print("SMS类别:", sorted(alert_classes_2))
                    print("安全类别:", sorted(SAFETY_CLASSES))
                    print("最终YOLOE类别:", final_classes)
                    print("=" * 60)
                    try:
                        model.set_classes(final_classes)
                        current_classes = final_classes.copy()
                        print("[YOLO] set_classes 成功:", current_classes)
                    except Exception as e:
                        print("[YOLO] set_classes 失败:", e)
                        traceback.print_exc()

            try:
                no_alert_classes = get_no_alert_classes() or set()
            except Exception as e:
                print("[NO ALERT ERROR]", e)
                no_alert_classes = set()

            res = model.predict(frame, conf=0.2, verbose=False)[0]
            all_objects = []

            for box in res.boxes:
                cls_id = int(box.cls[0])
                name = res.names.get(cls_id, str(cls_id))
                if name not in current_classes:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                obj = {"name": name, "conf": conf, "box": [x1, y1, x2, y2]}
                all_objects.append(obj)
                if name in alert_classes_2 and SMS_CONFIG.get("contact_phone"):
                    if alert_center.should_send_sms(name):
                        phone = SMS_CONFIG["contact_phone"]

                        ok = send_sms(
                            phone=phone,
                            message=name,                 # → alert_type
                            location=location,             # → patient_name
                            conf=obj.get("conf")           # → value
                        )

                        if not ok:
                            alert_center.rollback(name)



            print("[DEBUG] all_objects names:", [o["name"] for o in all_objects])
            print("[DEBUG] current_classes:", current_classes)
            print("[DEBUG] SMS alert_classes:", sorted(alert_classes_2))

            current_detected_indices = []
            if need_person_alert and any(o["name"] == "person" for o in all_objects):
                if "person" in alert_classes:
                    idx = alert_classes.index("person")
                    if idx not in current_detected_indices:
                        current_detected_indices.append(idx)

            abnormal = False
            abnormal_conf = 0.0
            person_obj = None

            if need_helmet_check:
                single_frame_abnormal, person_obj = check_no_hat(all_objects)
                if person_obj:
                    x1, y1, x2, y2 = person_obj["box"]
                    pid = (int((x1 + x2) / 2) // 30, int((y1 + (y2 - y1) * 0.2) // 30))
                    if single_frame_abnormal:
                        if pid not in no_helmet_timer:
                            no_helmet_timer[pid] = now
                        if now - no_helmet_timer[pid] >= NO_HELMET_ALERT_SECONDS:
                            abnormal = True
                            abnormal_conf = float(person_obj["conf"])
                    else:
                        no_helmet_timer.pop(pid, None)
                no_helmet_timer = {k: v for k, v in no_helmet_timer.items() if now - v < 10}

                if abnormal and "person without helmet" in alert_classes:
                    idx = alert_classes.index("person without helmet")
                    if idx not in current_detected_indices:
                        current_detected_indices.append(idx)

            update_detected_indices_to_server(current_detected_indices)
            if current_detected_indices:
                print("[ALERT] 检测到告警对象索引:", current_detected_indices)
                print("[ALERT] 对应对象:", [alert_classes[i] for i in current_detected_indices])

            vis = frame.copy()
            for obj in all_objects:
                x1, y1, x2, y2 = obj["box"]
                name = obj["name"]
                color = (0, 165, 255) if name == "helmet" else (255, 0, 0) if name in no_alert_classes else (0, 255, 0)
                if abnormal and name == "person" and obj is person_obj:
                    color = (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {obj['conf']:.2f}"
                if name in no_alert_classes:
                    label += " [NO ALERT]"
                cv2.putText(vis, label, (x1, max(y1 - 6, 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(vis, "Detecting: " + ", ".join(current_classes[:8]),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if abnormal:
                cv2.putText(vis, "WARNING: No Helmet!", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            save = save_abnormal = False
            if all_objects:
                if abnormal:
                    if now - last_abnormal > abnormal_interval:
                        save_abnormal, last_abnormal = True, now
                else:
                    if now - last_save > save_interval:
                        save, last_save = True, now

            if save or save_abnormal:
                frame_time = datetime.datetime.now()
                prefix = "abnormal_" if save_abnormal else "normal_"
                filename = f"{prefix}{current_camera_id}_{int(frame_time.timestamp()*1000)}.jpg"
                path = os.path.join(SAVE_DIR, filename)
                if cv2.imwrite(path, vis):
                    db_queue.put({
                        "camera_id": current_camera_id,
                        "image_path": path,
                        "frame_time": frame_time,
                        "objects": all_objects,
                        "abnormal": abnormal,
                        "abnormal_conf": abnormal_conf,
                        "location": location,
                        "camera_type": camera_type
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

if __name__ == "__main__":
    main()