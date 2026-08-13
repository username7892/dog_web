from web_server_2 import update_frame, start_web, get_display_classes  
import csv
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
    #"rtsp://100.95.170.3:8555/camera-rear"
    "rtsp://10.21.34.104:8555/camera-rear"
)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_PATH = "result.csv"
CSV_HEADERS = ["ID", "检测事件", "记录时间", "地点位置", "是否异常", "前/后摄"]

# 摄像头信息集中配置，后续新增摄像头时只需要扩展映射。
CAMERA_LOCATIONS = {
    "rear_camera": "后摄像头",
}
CAMERA_TYPES = {
    #"cam01": "前摄",
    "rear_camera": "后摄"
}

# 数据库队列：只传“元信息 + 文件路径”
db_queue = queue.Queue(maxsize=200)
# CSV 队列：一张已保存图片只投递一条事件记录。
csv_queue = queue.Queue(maxsize=200)


def get_camera_info(camera_id):
    """返回摄像头的位置和类型；未知编号保留可识别的默认值。"""
    location = CAMERA_LOCATIONS.get(camera_id, camera_id)
    camera_type = CAMERA_TYPES.get(camera_id, "前摄")
    return location, camera_type


def get_event_name(objects, abnormal):
    """异常事件优先；正常事件合并同一帧内出现的不同类别。"""
    if abnormal:
        return "未佩戴安全帽"

    class_names = list(dict.fromkeys(obj["name"] for obj in objects))
    return ",".join(class_names) if class_names else "未检测到目标"


def get_next_event_id(csv_path):
    """从已有 CSV 恢复自增 ID，避免程序重启后产生重复编号。"""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 1

    last_id = 0
    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
            for row in csv.DictReader(csv_file):
                try:
                    last_id = max(last_id, int(row.get("ID", 0)))
                except (TypeError, ValueError):
                    continue
    except (OSError, csv.Error) as exc:
        print("[CSV READ ERROR]", exc)
    return last_id + 1

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
# CSV 线程（实时追加）
# ======================

def csv_worker(csv_path=CSV_PATH):
    """实时追加事件；仅当文件不存在或为空时写入表头。"""
    print("[CSV] CSV线程启动")

    # 程序启动即创建 CSV；没有检测记录时也会保留完整表头。
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as csv_file:
                csv.writer(csv_file).writerow(CSV_HEADERS)
    except (OSError, csv.Error) as exc:
        print("[CSV INIT ERROR]", exc)

    while True:
        event = csv_queue.get()
        try:
            write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.writer(csv_file)
                if write_header:
                    writer.writerow(CSV_HEADERS)
                writer.writerow([
                    event["id"],
                    event["event"],
                    event["time"].strftime("%Y-%m-%d %H:%M:%S"),
                    event["location"],
                    "是" if event["abnormal"] else "否",
                    event["camera_type"],
                ])
                csv_file.flush()
        except (OSError, csv.Error, KeyError) as exc:
            print("[CSV ERROR]", exc)
        finally:
            csv_queue.task_done()


# ======================
# 主逻辑
# ======================

def main():
    camera_id = "rear_camera"
    location, camera_type = get_camera_info(camera_id)
    next_event_id = get_next_event_id(CSV_PATH)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = YOLOE("yoloe-v8s-seg.pt")
    model.to(device)
    model.set_classes(["person", "helmet","car", "dog","cat", "bicycle", 
                       "motorbike", "bus", "truck","traffic light", "fire hydrant", 
                       "stop sign", "parking meter", "bench", "bird", "horse", "sheep", 
                       "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
                       "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", 
                       "snowboard", "sports ball", "kite", "baseball bat", 
                       "baseball glove", "skateboard", "surfboard", "tennis racket", 
                       "bottle", "wine glass", "cup", "fork", "knife", "spoon","bowl",
                       "banana","apple","sandwich","orange","broccoli","carrot","hot dog",
                       "pizza","donut","cake","chair","couch","potted plant","bed",
                       "dining table","toilet","tv","laptop","mouse","remote","keyboard",
                       "cell phone","microwave","oven","toaster","sink","refrigerator",
                       "book","clock","vase","scissors","teddy bear","hair drier",
                       "toothbrush","hair brush","hair dryer","tooth brush","fire extinguisher","ladder","stool","barrel","basket","bucket","bench",
                       "crate","pillow","mirror","rug","curtain","blinds","fan",
                       "lamp","candle","vase","bottle","glass","cup","plate","bowl",])

    buffer = RingBuffer(50)
    rtsp = DualRTSP(RTSP_URL, buffer)
    threading.Thread(target=rtsp.run, daemon=True).start()

    threading.Thread(target=start_web, daemon=True).start()
    threading.Thread(target=database_worker, daemon=True).start()
    threading.Thread(target=csv_worker, daemon=True).start()

    time.sleep(3)

    delay = 20
    last_save = 0
    last_abnormal = 0
    save_interval = 1
    abnormal_interval = 10

    # ========== 导入前端显示类别函数 ==========

    while True:
        frame = buffer.get(delay)
        if frame is None:
            time.sleep(0.01)
            continue

        # ---------- YOLO全量识别（不变） ----------
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

            # ========== 修改：绘制两套框 ==========
            # 1. 全量框 - 用于保存图片（数据库用）
            full_vis = frame.copy()
            for obj in objects:
                x1, y1, x2, y2 = obj["box"]
                cv2.rectangle(full_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(full_vis, obj["name"], (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 2. 筛选框 - 用于前端显示（只框选中的类别）
            display_classes = get_display_classes()  # 从web_server_2获取
            
            display_vis = frame.copy()
            for obj in objects:
                # 如果display_classes为空，显示全部；否则只显示选中的
                if display_classes and obj["name"] not in display_classes:
                    continue
                    
                x1, y1, x2, y2 = obj["box"]
                color = (0, 0, 255) if obj["name"] == "helmet" else (0, 255, 0)
                cv2.rectangle(display_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_vis, obj["name"], (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # ---------- 保存图片用全量框 ----------
            now = time.time()
            save = False
            save_abnormal = False

            if abnormal:
                if now - last_abnormal > abnormal_interval:
                    save_abnormal = True
                    last_abnormal = now
            elif now - last_save > save_interval:
                save = True
                last_save = now

            if save or save_abnormal:
                frame_time = datetime.datetime.now()
                filename = f"{camera_id}_{int(frame_time.timestamp()*1000)}.jpg"
                path = os.path.join(SAVE_DIR, filename)
                # 保存全量框的图片
                image_saved = cv2.imwrite(path, full_vis)

                if image_saved:
                    frame_data = {
                        "camera_id": camera_id,
                        "image_path": path,
                        "frame_time": frame_time,
                        "objects": objects,
                        "abnormal": abnormal,
                        "abnormal_conf": abnormal_conf,
                        "event_id": next_event_id,
                        "location": location,
                        "camera_type": camera_type,
                    }
                    event_data = {
                        "id": frame_data["event_id"],
                        "event": get_event_name(objects, abnormal),
                        "time": frame_data["frame_time"],
                        "location": frame_data["location"],
                        "abnormal": frame_data["abnormal"],
                        "camera_type": frame_data["camera_type"],
                    }

                    db_queue.put(frame_data)
                    csv_queue.put(event_data)
                    next_event_id += 1
                else:
                    print("[IMAGE SAVE ERROR]", path)

            # ---------- 前端显示用筛选框 ----------
            update_frame(display_vis)  # 传给前端的是只框选中类别的帧
            cv2.imshow("dual RTSP", display_vis)  # 本地预览也用筛选框（可选）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print("[MAIN ERROR]", e)

    rtsp.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
