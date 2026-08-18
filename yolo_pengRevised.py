from web_server import (
    update_frame,
    start_web,
    get_display_classes,
    get_current_classes,
    get_no_alert_classes
)

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
    #"rtsp://m20-detector:f715e51840a1359d569bbb9a42af402e@120.26.18.138:8554/camera-front"
    "rtsp://admin:dhlb839.@192.168.50.64:554/Streaming/Channels/101"
)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

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

def check_no_hat(objects):
    persons = [o for o in objects if o["name"] == "person"]
    helmets = [o for o in objects if o["name"] == "helmet"]

    if not persons or not helmets:
        return False, None

    for person in persons:
        px1, py1, px2, py2 = person["box"]
        person_height = py2 - py1
        head_y1 = py1
        head_y2 = py1 + person_height * 0.3

        has_helmet = False
        for helmet in helmets:
            hx1, hy1, hx2, hy2 = helmet["box"]
            helmet_cx = (hx1 + hx2) / 2
            helmet_cy = (hy1 + hy2) / 2

            if px1 < helmet_cx < px2 and head_y1 < helmet_cy < head_y2:
                overlap_x1 = max(px1, hx1)
                overlap_y1 = max(head_y1, hy1)
                overlap_x2 = min(px2, hx2)
                overlap_y2 = min(head_y2, hy2)

                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    head_area = (px2 - px1) * (head_y2 - head_y1)

                    if head_area > 0 and overlap_area / head_area > 0.2:
                        has_helmet = True
                        break

        if not has_helmet:
            return True, person

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
    model = YOLOE("yoloe-v8s-seg.pt")
    model.to(device)
    print("[YOLO] YOLOE 加载完成")

    # 初始化检测类别
    current_classes = ["person", "car", "dog", "cup", "phone"]
    current_classes = list(dict.fromkeys(current_classes))
    print(f"\n{'='*32}\n【INIT】初始检测类别:\n{current_classes}\n{'='*32}\n")

    try:
        model.set_classes(current_classes)
        print("[YOLO] 初始类别设置成功:", current_classes)
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

    # 主循环
    while True:
        frame = buffer.get(delay)
        if frame is None:
            time.sleep(0.01)
            continue

        try:
            now = time.time()

            # ① 从 Flask 获取最新的检测类别
            if now - last_class_check >= class_update_interval:
                last_class_check = now
                try:
                    new_classes = get_current_classes()
                except Exception as e:
                    print("[CLASS ERROR]", e)
                    new_classes = current_classes

                if new_classes is None:
                    new_classes = []

                new_classes = [str(c).strip() for c in new_classes if str(c).strip()]
                new_classes = list(dict.fromkeys(new_classes))

                if new_classes != current_classes:
                    print(f"\n{'='*32}\n【YOLO CLASS UPDATE】\n旧类别: {current_classes}\n新类别: {new_classes}\n{'='*32}\n")
                    current_classes = new_classes.copy()
                    if current_classes:
                        try:
                            model.set_classes(current_classes)
                            print("[YOLO] set_classes 成功:", current_classes)
                        except Exception as e:
                            print("[YOLO] set_classes 失败:", e)
                            traceback.print_exc()
                    else:
                        print("[YOLO] 检测类别为空，跳过 set_classes")

            # ② 获取不报警对象
            try:
                no_alert_classes = get_no_alert_classes()
            except Exception as e:
                print("[NO ALERT ERROR]", e)
                no_alert_classes = set()

            if no_alert_classes is None:
                no_alert_classes = set()
            no_alert_classes = set(no_alert_classes)

            # ③ 如果没有检测类别，跳过推理
            if not current_classes:
                vis = frame.copy()
                cv2.putText(vis, "No detection classes selected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                cv2.putText(vis, "Please set objects via web UI", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                update_frame(vis)
                cv2.imshow("YOLOE Front Camera", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.03)
                continue

            # ④ YOLOE 推理
            res = model.predict(frame, conf=0.25, verbose=False)[0]

            # ⑤ 获取检测结果
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

            # ⑥ 安全帽检测
            abnormal = False
            abnormal_conf = 0
            person_can_alert = "person" not in no_alert_classes

            if "person" in current_classes and "helmet" in current_classes and person_can_alert:
                abnormal, person_obj = check_no_hat(all_objects)
                if abnormal and person_obj:
                    abnormal_conf = person_obj["conf"]

            # ⑦ 绘制检测框
            vis = frame.copy()
            for obj in all_objects:
                x1, y1, x2, y2 = obj["box"]
                is_no_alert = obj["name"] in no_alert_classes

                if obj["name"] == "helmet":
                    color = (0, 165, 255)
                elif is_no_alert:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{obj['name']} {obj['conf']:.2f}" + (" [NO ALERT]" if is_no_alert else "")
                cv2.putText(vis, label, (x1, max(y1 - 6, 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ⑧ 显示当前检测类别
            classes_text = "Detecting: " + ", ".join(current_classes[:5])
            if len(current_classes) > 5:
                classes_text += f" (+{len(current_classes)-5} more)"
            cv2.putText(vis, classes_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ⑨ 安全帽报警显示
            if abnormal:
                cv2.putText(vis, "WARNING: No Helmet!", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # ⑩ 保存图片
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

            # ⑪ 写入数据库
            if save or save_abnormal:
                frame_time = datetime.datetime.now()
                prefix = "abnormal_" if save_abnormal else "normal_"
                filename = f"{prefix}{camera_id}_{int(frame_time.timestamp() * 1000)}.jpg"
                path = os.path.join(SAVE_DIR, filename)
                image_saved = cv2.imwrite(path, vis)

                if image_saved:
                    frame_data = {
                        "camera_id": camera_id,
                        "image_path": path,
                        "frame_time": frame_time,
                        "objects": all_objects,
                        "abnormal": abnormal,
                        "abnormal_conf": abnormal_conf,
                        "location": location,
                        "camera_type": camera_type,
                    }

                    db_queue.put(frame_data)

                    if save_abnormal:
                        print("[ALERT] 检测到未佩戴安全帽！")
                        print("图片:", path)
                else:
                    print("[IMAGE SAVE ERROR]", path)

            # ⑫ 发送给 Web
            update_frame(vis)

            # 本地窗口
            cv2.imshow("YOLOE Front Camera", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as e:
            print("[MAIN ERROR]", e)
            traceback.print_exc()

    rtsp.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()