from web_server_2 import (
    update_frame,
    start_web,
    get_display_classes,
    get_current_classes,
    get_no_alert_classes,
    SMS_CONFIG,
)

import target_detection
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
from alert_center import AlertCenter, send_sms


# ============================================================
# 全局配置
# ============================================================

camera_id = "rear_camera"

RTSP_URL = (
    "rtsp://admin:dhlb839.@192.168.50.64:554/Streaming/Channels/102"
)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# 永远需要检测的类别
SAFETY_CLASSES = set()

# 图片保存目录
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 摄像头信息
CAMERA_LOCATIONS = {
    "rear_camera": "后摄像头",
}

CAMERA_TYPES = {
    "rear_camera": "rear camera",
}

# 数据库队列
db_queue = queue.Queue(maxsize=200)

# 告警中心
alert_center = AlertCenter(cooldown_seconds=300)


# ============================================================
# 摄像头信息
# ============================================================

def get_camera_info(camera_id):
    location = CAMERA_LOCATIONS.get(camera_id, camera_id)
    camera_type = CAMERA_TYPES.get(camera_id, "前摄")
    return location, camera_type


# ============================================================
# 类别名称标准化
# ============================================================

def norm_name(x):
    return str(x).strip().lower()


# ============================================================
# FrameBuffer
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
            if self._frame is None:
                return None
            return self._frame.copy()


# ============================================================
# 双 RTSP 自动切换
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
                continue
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
        print(f"[RTSP] 当前切换到 cap{self.active}")

    def stop(self):
        self.running = False
        for cap in (self.cap1, self.cap2):
            try:
                if cap:
                    cap.release()
            except Exception:
                pass


# ============================================================
# 推送告警索引到 Web Server
# ============================================================

def update_detected_indices_to_server(indices):
    try:
        print(f"[YOLO] 准备推送告警索引: {indices}")
        response = requests.post(
            "http://127.0.0.1:5001/api/update_detected_indices",
            json={"indices": indices},
            timeout=0.5
        )
        print(f"[YOLO] 推送结果: {response.status_code}")
    except Exception as e:
        print(f"[YOLO] 告警索引推送失败: {e}")


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
            cid = data["camera_id"]
            cursor.execute(
                "SELECT COALESCE(MAX(camera_seq), -1) + 1 FROM images WHERE camera_id = %s",
                (cid,)
            )
            seq = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO images (camera_id, camera_seq, image, frame_time) VALUES (%s, %s, %s, %s)",
                (cid, seq, data["image_path"], data["frame_time"])
            )
            image_id = cursor.lastrowid
            for obj in data["objects"]:
                x1, y1, x2, y2 = obj["box"]
                cursor.execute(
                    "INSERT INTO detections (image_id, class_name, confidence, x1, y1, x2, y2) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (image_id, obj["name"], obj["conf"], x1, y1, x2, y2)
                )
            if data["abnormal"]:
                cursor.execute(
                    "INSERT INTO abnormal_events (image_id, event_type, description, camera_id, confidence) VALUES (%s, %s, %s, %s, %s)",
                    (image_id, "no_hat", "人员未佩戴安全帽", cid, data["abnormal_conf"])
                )
            db.commit()
            print(f"[DB] 保存成功 image_id={image_id}, camera_id={cid}, camera_seq={seq}")
        except Exception as e:
            print("[DB ERROR]", e)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db_queue.task_done()


# ============================================================
# 数据库队列
# ============================================================

def put_db_queue(data):
    try:
        db_queue.put_nowait(data)
    except queue.Full:
        print("[DB] 队列已满，本次数据库记录跳过")


# ============================================================
# 绘制检测框
# ============================================================

def draw_detections(frame, objects, abnormal, no_alert_classes, current_classes):
    vis = frame.copy()
    no_alert_set = {norm_name(c) for c in no_alert_classes}
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        name = obj["name"]
        conf = obj["conf"]
        name_norm = norm_name(name)
        is_no_alert = name_norm in no_alert_set
        if abnormal and name_norm == "person":
            color = (0, 0, 255)
        elif name_norm == "helmet":
            color = (255, 0, 0)
        elif is_no_alert:
            color = (255, 0, 255)
        elif name_norm == "person":
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)
        thickness = 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = f"{name} {conf:.2f}"
        if is_no_alert:
            label += " [NO ALERT]"
        text_y = max(y1 - 8, 20)
        cv2.putText(vis, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    detecting_text = "Detecting: " + ", ".join(current_classes[:8])
    cv2.putText(vis, detecting_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if abnormal:
        cv2.putText(vis, "WARNING: NO HELMET", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
    return vis


# ============================================================
# 主程序
# ============================================================

def main():
    current_camera_id = "rear_camera"
    location, camera_type = get_camera_info(current_camera_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print(f"[SYSTEM] device: {device}")
    print(f"[SYSTEM] camera: {current_camera_id}")
    print(f"[SYSTEM] location: {location}")
    print(f"[SYSTEM] camera_type: {camera_type}")
    print("=" * 60)

    print("[YOLO] 正在加载 YOLOE...")
    model = YOLOE("yoloe-v8l-seg.pt").to(device)
    print("[YOLO] YOLOE 加载完成")

    current_classes = []
    last_yolo_classes = []
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
    class_update_interval = 0.5
    last_class_check = 0
    target_fps = 15
    frame_interval = 1.0 / target_fps
    last_infer_time = 0
    no_helmet_timer = {}
    NO_HELMET_ALERT_SECONDS = 2.0
    last_vis_frame = None
    no_alert_classes_raw = set()
    sms_alert_classes = set()
    alert_classes = []

    print("[MAIN] 检测线程启动")

    while True:
        try:
            frame = buffer.get()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()

            # 获取声音告警配置
            try:
                response = requests.get("http://127.0.0.1:5001/api/get_audio_alert_config", timeout=0.5)
                if response.status_code == 200:
                    alert_config = response.json()
                else:
                    alert_config = {"classes": []}
            except Exception:
                alert_config = {"classes": []}

            alert_classes = [str(c).strip() for c in alert_config.get("classes", []) if str(c).strip()]
            alert_classes = list(dict.fromkeys(alert_classes))
            need_person_alert = "person" in alert_classes
            need_helmet_check = "person without helmet" in alert_classes

            # 每 0.5 秒更新一次类别
            if now - last_class_check >= class_update_interval:
                last_class_check = now

                # 1. 获取前端检测类别
                try:
                    frontend_classes = get_current_classes() or []
                except Exception as e:
                    print("[CLASS ERROR]", e)
                    frontend_classes = []
                frontend_classes = [str(c).strip() for c in frontend_classes if str(c).strip()]
                frontend_classes = list(dict.fromkeys(frontend_classes))

                # 2. 获取无需告警类别
                try:
                    no_alert_classes = get_no_alert_classes() or []
                except Exception as e:
                    print("[NO ALERT ERROR]", e)
                    no_alert_classes = set()
                no_alert_classes = {str(c).strip() for c in no_alert_classes if str(c).strip()}
                no_alert_classes_raw = set(no_alert_classes)

                # 3. SMS 类别
                alert_str_2 = SMS_CONFIG.get("alert_object", "")
                if alert_str_2:
                    sms_alert_classes = {x.strip() for x in alert_str_2.split(",") if x.strip()}
                else:
                    sms_alert_classes = set()

                # 4. 构造原始来源对象
                raw_source_objects = set(frontend_classes) | set(no_alert_classes) | sms_alert_classes

                # 5. 声音告警 person
                if need_person_alert:
                    raw_source_objects.add("person")

                # 6. 声音告警 person without helmet
                if need_helmet_check:
                    raw_source_objects.add("person without helmet")

                # 7. 显示类别
                try:
                    display_classes = get_display_classes() or []
                except Exception as e:
                    print("[DISPLAY CLASS ERROR]", e)
                    display_classes = []
                display_classes = [str(c).strip() for c in display_classes if str(c).strip()]
                display_classes = list(dict.fromkeys(display_classes))
                raw_source_objects |= set(display_classes)

                # 8. 使用 CustomSplitRule
                try:
                    final_classes = sorted(
                        target_detection.CustomSplitRule.split_all(raw_source_objects)
                        | SAFETY_CLASSES
                    )
                except Exception as e:
                    print("[CLASS] CustomSplitRule.split_all 失败:", e)
                    traceback.print_exc()
                    final_classes = sorted(raw_source_objects | SAFETY_CLASSES)

                # 打印类别信息
                print("\n" + "=" * 60)
                print("[CLASS UPDATE]")
                print("[CLASS] frontend:", frontend_classes)
                print("[CLASS] no alert:", sorted(no_alert_classes))
                print("[CLASS] SMS:", sorted(sms_alert_classes))
                print("[CLASS] audio alert:", alert_classes)
                print("[CLASS] display:", display_classes)
                print("[CLASS] raw source:", sorted(raw_source_objects))
                print("[CLASS] SAFETY:", sorted(SAFETY_CLASSES))
                print("[CLASS] YOLO classes:", final_classes)
                print("=" * 60)

                # 9. 更新 YOLOE 类别
                if final_classes:
                    model.set_classes(final_classes)
                    current_classes = final_classes.copy()
                    last_yolo_classes = final_classes.copy()
                    print("[YOLO] set_classes 成功:", current_classes)
                else:
                    current_classes = []
                    last_yolo_classes = []
                    print("[YOLO] 当前没有需要检测的类别")

            # 当前没有类别
            if not current_classes:
                update_frame(frame)
                time.sleep(0.001)
                continue

            # 控制 YOLO 推理 FPS
            if now - last_infer_time < frame_interval:
                if last_vis_frame is not None:
                    update_frame(last_vis_frame)
                else:
                    update_frame(frame)
                continue

            last_infer_time = now

            # YOLO 推理
            results = model.predict(frame, conf=0.5, verbose=False)
            if not results:
                update_frame(frame)
                continue

            res = results[0]
            all_objects = []

            for box in res.boxes:
                cls_id = int(box.cls[0])
                name = res.names.get(cls_id, str(cls_id))
                if name not in current_classes:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                all_objects.append({"name": name, "conf": conf, "box": [x1, y1, x2, y2]})

            # DEBUG
            print("[DEBUG] objects:", [o["name"] for o in all_objects])
            print("[DEBUG] current_classes:", current_classes)
            print("[DEBUG] no_alert_classes:", no_alert_classes_raw)
            print("[DEBUG] sms_alert_classes:", sms_alert_classes)

            # SMS 告警
            for obj in all_objects:
                obj_name = norm_name(obj["name"])
                if obj_name in {norm_name(x) for x in sms_alert_classes} and SMS_CONFIG.get("contact_phone"):
                    if alert_center.should_send_sms(obj["name"]):
                        phone = SMS_CONFIG["contact_phone"]
                        ok = send_sms(phone=phone, message=obj["name"], location=location, conf=obj.get("conf"))
                        if not ok:
                            alert_center.rollback(obj["name"])

            # 当前检测到的声音告警索引
            current_detected_indices = []

            # person 告警
            if need_person_alert and any(norm_name(o["name"]) == "person" for o in all_objects) and "person" in alert_classes:
                idx = alert_classes.index("person")
                if idx not in current_detected_indices:
                    current_detected_indices.append(idx)

            # 安全帽检测
            abnormal = False
            abnormal_conf = 0.0
            person_obj = None

            if any(norm_name(o["name"]) == "person" for o in all_objects) and "person without helmet" in raw_source_objects:
                try:
                    single_frame_abnormal, person_obj = target_detection.HelmetDetector.check_no_hat(all_objects)
                except Exception as e:
                    print("[HELMET] HelmetDetector.check_no_hat 失败:", e)
                    traceback.print_exc()
                    single_frame_abnormal = False
                    person_obj = None

                if person_obj:
                    x1, y1, x2, y2 = person_obj["box"]
                    head_cx = int((x1 + x2) / 2)
                    head_cy = int(y1 + (y2 - y1) * 0.2)
                    pid = (head_cx // 30, head_cy // 30)

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

            # 推送声音告警索引
            update_detected_indices_to_server(current_detected_indices)

            if current_detected_indices:
                print("[ALERT] 检测到告警对象索引:", current_detected_indices)

            # 绘制检测结果
            vis = draw_detections(frame, all_objects, abnormal, no_alert_classes_raw, current_classes)
            last_vis_frame = vis

            # 图片保存
            save = False
            save_abnormal = False

            if all_objects:
                if abnormal:
                    if now - last_abnormal >= abnormal_interval:
                        save_abnormal = True
                        last_abnormal = now
                else:
                    if now - last_save >= save_interval:
                        save = True
                        last_save = now

            if save or save_abnormal:
                frame_time = datetime.datetime.now()
                prefix = "abnormal_" if save_abnormal else "normal_"
                filename = f"{prefix}{current_camera_id}_{int(frame_time.timestamp() * 1000)}.jpg"
                path = os.path.join(SAVE_DIR, filename)
                if cv2.imwrite(path, vis):
                    print("[IMAGE] 保存成功:", path)
                    put_db_queue({
                        "camera_id": current_camera_id,
                        "image_path": path,
                        "frame_time": frame_time,
                        "objects": all_objects,
                        "abnormal": abnormal,
                        "abnormal_conf": abnormal_conf,
                        "location": location,
                        "camera_type": camera_type,
                    })
                else:
                    print("[IMAGE SAVE ERROR]", path)

            # 更新 Web 视频
            update_frame(vis)

            # 本地 OpenCV 窗口
            try:
                cv2.imshow("rear RTSP", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                pass

        except Exception as e:
            print("\n[MAIN ERROR]", e)
            traceback.print_exc()
            try:
                if frame is not None:
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


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()