from web_server import update_frame, start_web
import queue
import pymysql
import cv2
import os
import time
import threading
import torch
import datetime

from ultralytics import YOLOE

# ======================
# 全局配置
# ======================

RTSP_URL = (
    "rtsp://100.95.170.3:8555/camera-front"
)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

# 数据库队列：只传“元信息 + 文件路径”
db_queue = queue.Queue(maxsize=200)

# ======================
# RTSP 双缓冲（你原来的，不动）
# ======================

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
                if self.active == 1:
                    self.cap1.release()
                    self.cap1 = None
                    self.fixing = 1
                    threading.Thread(target=self._fix_cap, args=(1,), daemon=True).start()
                else:
                    self.cap2.release()
                    self.cap2 = None
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


# ======================
# 安全帽逻辑
# ======================

def check_no_hat(objects):
    persons = [o for o in objects if o["name"] == "person"]
    helmets = [o for o in objects if o["name"] == "helmet"]

    for p in persons:
        px1, py1, px2, py2 = p["box"]
        for h in helmets:
            hx1, hy1, hx2, hy2 = h["box"]
            cx = (hx1 + hx2) / 2
            cy = (hy1 + hy2) / 2
            if px1 < cx < px2 and py1 < cy < py1 + (py2 - py1) * 0.35:
                return False, None
    return True, persons[0] if persons else None


# ======================
# 数据库线程（只写 DB）
# ======================

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
            # 存图片路径
            cursor.execute(
                """
                INSERT INTO images (camera_id, image, frame_time)
                VALUES (%s,%s,%s)
                """,
                (data["camera_id"], data["image_path"], data["frame_time"])
            )
            image_id = cursor.lastrowid

            # 存检测框
            for obj in data["objects"]:
                x1, y1, x2, y2 = obj["box"]
                cursor.execute(
                    """
                    INSERT INTO detections
                    (image_id, class_name, confidence, x1, y1, x2, y2)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (image_id, obj["name"], obj["conf"], x1, y1, x2, y2)
                )

            # 异常事件
            if data["abnormal"]:
                cursor.execute(
                    """
                    INSERT INTO abnormal_events
                    (image_id, event_type, description, camera_id, confidence)
                    VALUES (%s,%s,%s,%s,%s)
                    """,
                    (image_id, "no_hat", "人员未佩戴安全帽",
                     data["camera_id"], data["abnormal_conf"])
                )

            db.commit()

        except Exception as e:
            print("[DB ERROR]", e)
            db.rollback()

        finally:
            db_queue.task_done()


# ======================
# 主逻辑
# ======================

def main():
    camera_id = "cam01"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = YOLOE("yoloe-v8s-seg.pt")
    model.to(device)
    model.set_classes(["person", "helmet"])

    buffer = RingBuffer(50)
    rtsp = DualRTSP(RTSP_URL, buffer)
    threading.Thread(target=rtsp.run, daemon=True).start()

    threading.Thread(target=start_web, daemon=True).start()
    threading.Thread(target=database_worker, daemon=True).start()

    time.sleep(3)

    delay = 20
    last_save = 0
    last_abnormal = 0
    save_interval = 1
    abnormal_interval = 10

    while True:
        frame = buffer.get(delay)
        if frame is None:
            time.sleep(0.01)
            continue
            
        # frame = cv2.resize(frame, (640, 360),
        # interpolation=cv2.INTER_AREA
        # )

        # ---------- YOLO ----------
        try:
            res = model.predict(frame, conf=0.25, verbose=False)[0]

            objects = []
            for box in res.boxes:
                cls_id = int(box.cls[0])
                name = res.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                objects.append({
                    "name": name,
                    "conf": conf,
                    "box": [x1, y1, x2, y2]
                })

            abnormal, person = check_no_hat(objects)
            abnormal_conf = person["conf"] if abnormal and person else 0

            # ---------- 画框 ----------
            vis = frame.copy()
            for obj in objects:
                x1, y1, x2, y2 = obj["box"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, obj["name"], (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # ---------- 是否存 ----------
            now = time.time()
            save = False
            save_abnormal = False

            if now - last_save > save_interval:
                save = True
                last_save = now

            if abnormal and now - last_abnormal > abnormal_interval:
                save_abnormal = True
                last_abnormal = now

            if save or save_abnormal:
                filename = f"{camera_id}_{int(time.time()*1000)}.jpg"
                path = os.path.join(SAVE_DIR, filename)
                cv2.imwrite(path, vis)

                db_queue.put({
                    "camera_id": camera_id,
                    "image_path": path,
                    "frame_time": datetime.datetime.now(),
                    "objects": objects,
                    "abnormal": abnormal,
                    "abnormal_conf": abnormal_conf
                })

            # ---------- Web ----------
            update_frame(vis)
            cv2.imshow("dual RTSP", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print("[MAIN ERROR]", e)

    rtsp.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
