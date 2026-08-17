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
import traceback

from ultralytics import YOLOE


# ============================================================
# 全局配置
# ============================================================

RTSP_URL = (
    "rtsp://m20-detector:f715e51840a1359d569bbb9a42af402e@120.26.18.138:8554/camera-rear"
    #"rtsp://10.21.34.104:8555/camera-rear"
)

# OpenCV 使用 FFmpeg，并强制 RTSP TCP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# 图片保存目录
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

# CSV
CSV_PATH = "result.csv"
CSV_HEADERS = [
    "ID",
    "检测事件",
    "记录时间",
    "地点位置",
    "是否异常",
    "前/后摄"
]

# 摄像头配置
CAMERA_LOCATIONS = {
    "rear_camera": "后摄像头",
}

CAMERA_TYPES = {
    "rear_camera": "rear camera"
}

# 数据库队列
db_queue = queue.Queue(maxsize=200)

# CSV 队列
csv_queue = queue.Queue(maxsize=200)


# ============================================================
# 摄像头信息
# ============================================================

def get_camera_info(camera_id):
    location = CAMERA_LOCATIONS.get(camera_id, camera_id)
    camera_type = CAMERA_TYPES.get(camera_id, "前摄")
    return location, camera_type


# ============================================================
# 事件名称
# ============================================================

def get_event_name(objects, abnormal):
    """异常事件优先，正常情况下合并所有检测类别"""
    if abnormal:
        return "未佩戴安全帽"

    class_names = list(
        dict.fromkeys(obj["name"] for obj in objects)
    )

    if class_names:
        return ",".join(class_names)

    return "未检测到目标"


# ============================================================
# CSV 自增 ID
# ============================================================

def get_next_event_id(csv_path):
    """从已有 CSV 恢复自增 ID"""
    if not os.path.exists(csv_path):
        return 1
    if os.path.getsize(csv_path) == 0:
        return 1

    last_id = 0
    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    last_id = max(last_id, int(row.get("ID", 0)))
                except (TypeError, ValueError):
                    continue
    except (OSError, csv.Error) as exc:
        print("[CSV READ ERROR]", exc)

    return last_id + 1


# ============================================================
# 简易帧缓冲（仅用于双路切换，不做延迟）
# ============================================================

class FrameBuffer:
    """线程安全的单帧缓冲，始终保存最新一帧"""

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
# 双 RTSP 连接（修复版）
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
        self._stop_event = threading.Event()

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
# 安全帽检测
# ============================================================

def check_no_hat(objects):
    """判断是否有未佩戴安全帽的人员"""
    persons = [o for o in objects if o["name"] == "person"]
    helmets = [o for o in objects if o["name"] == "helmet"]

    if not persons:
        return False, None
    if not helmets:
        return True, persons[0]

    for person in persons:
        px1, py1, px2, py2 = person["box"]
        head_y2 = py1 + (py2 - py1) * 0.35

        for helmet in helmets:
            hx1, hy1, hx2, hy2 = helmet["box"]
            cx = (hx1 + hx2) / 2
            cy = (hy1 + hy2) / 2

            if px1 < cx < px2 and py1 < cy < head_y2:
                return False, None

    return True, persons[0]


# ============================================================
# 数据库线程
# ============================================================

def database_worker():
    print("[DB] 数据库线程启动")
    try:
        db = pymysql.connect(
            host="localhost", user="root", password="123456",
            database="yolo_images", charset="utf8mb4", autocommit=False
        )
    except Exception as exc:
        print("[DB] 数据库连接失败:", exc)
        return

    cursor = db.cursor()

    while True:
        data = db_queue.get()
        try:
            cursor.execute(
                "INSERT INTO images (camera_id, image, frame_time) VALUES (%s,%s,%s)",
                (data["camera_id"], data["image_path"], data["frame_time"])
            )
            image_id = cursor.lastrowid

            for obj in data["objects"]:
                x1, y1, x2, y2 = obj["box"]
                cursor.execute(
                    """INSERT INTO detections
                       (image_id, class_name, confidence, x1, y1, x2, y2)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                    (image_id, obj["name"], obj["conf"], x1, y1, x2, y2)
                )

            if data["abnormal"]:
                cursor.execute(
                    """INSERT INTO abnormal_events
                       (image_id, event_type, description, camera_id, confidence)
                       VALUES (%s,%s,%s,%s,%s)""",
                    (image_id, "no_hat", "人员未佩戴安全帽",
                     data["camera_id"], data["abnormal_conf"])
                )

            db.commit()
            print(f"[DB] 保存成功 image_id={image_id}")

        except Exception as exc:
            print("[DB ERROR]", exc)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db_queue.task_done()


# ============================================================
# CSV 线程
# ============================================================

def csv_worker(csv_path=CSV_PATH):
    print("[CSV] CSV线程启动")

    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(CSV_HEADERS)
    except (OSError, csv.Error) as exc:
        print("[CSV INIT ERROR]", exc)

    while True:
        event = csv_queue.get()
        try:
            write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
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
                f.flush()
        except (OSError, csv.Error, KeyError) as exc:
            print("[CSV ERROR]", exc)
        finally:
            csv_queue.task_done()


# ============================================================
# 安全入队
# ============================================================

def put_db_queue(data):
    try:
        db_queue.put_nowait(data)
    except queue.Full:
        print("[DB] 队列已满，本次数据库记录跳过")


def put_csv_queue(data):
    try:
        csv_queue.put_nowait(data)
    except queue.Full:
        print("[CSV] 队列已满，本次 CSV 记录跳过")


# ============================================================
# 绘制检测框（独立函数，清晰可读）
# ============================================================

def draw_detections(frame, objects, abnormal):
    """在帧上绘制检测框和标签，返回绘制后的画面"""
    vis = frame.copy()

    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        name = obj["name"]
        conf = obj["conf"]

        # 颜色规则
        if name == "person" and abnormal:
            color = (0, 0, 255)       # 红：未戴安全帽的人
        elif name == "helmet":
            color = (255, 0, 0)       # 蓝：安全帽
        elif name == "person":
            color = (0, 255, 0)       # 绿：正常人员
        else:
            color = (0, 255, 255)      # 黄：其他物体

        # 画框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 标签
        label = f"{name} {conf:.2f}"
        text_y = max(y1 - 8, 20)
        cv2.putText(vis, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # 异常警告
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
    next_event_id = get_next_event_id(CSV_PATH)

    # --------------------------------------------------------
    # 设备选择
    # --------------------------------------------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 40)
    print(f"[SYSTEM] device: {device}")
    print(f"[SYSTEM] camera: {camera_id}")
    print(f"[SYSTEM] location: {location}")
    print(f"[SYSTEM] camera_type: {camera_type}")
    print("=" * 40)

    # --------------------------------------------------------
    # 加载 YOLOE 模型
    # --------------------------------------------------------
    print("[YOLO] 正在加载模型...")
    model = YOLOE("yoloe-v8s-seg.pt")
    model.to(device)
    print("[YOLO] 模型加载完成")

    # --------------------------------------------------------
    # 帧缓冲（不做延迟，始终取最新帧）
    # --------------------------------------------------------
    buffer = FrameBuffer()

    rtsp = DualRTSP(RTSP_URL, buffer)
    threading.Thread(target=rtsp.run, daemon=True).start()

    # Web 服务
    threading.Thread(target=start_web, daemon=True).start()

    # DB 线程
    threading.Thread(target=database_worker, daemon=True).start()

    # CSV 线程
    threading.Thread(target=csv_worker, daemon=True).start()

    time.sleep(3)

    # --------------------------------------------------------
    # 控制参数
    # --------------------------------------------------------
    last_save = 0
    last_abnormal = 0
    save_interval = 2        # 正常保存间隔(秒)
    abnormal_interval = 10   # 异常保存间隔(秒)
    last_classes = None

    # 帧率控制：约 15 FPS 推理（兼顾流畅与性能）
    target_fps = 15
    frame_interval = 1.0 / target_fps
    last_infer_time = 0
    last_detected_classes = None
    last_object_save_time = 0     # 普通物体保存计时
    last_abnormal_save_time = 0
    # 新增：记录已检测到的物体种类
    last_detected_classes = None

    # 持续显示上一次检测结果（关键！）
    last_objects = []
    last_abnormal = False
    last_vis_frame = None

    print("[MAIN] 检测线程启动")

    # ========================================================
    # 主循环
    # ========================================================
    while True:
        try:
            # ----------------------------------------------------
            # 取最新帧（无延迟）
            # ----------------------------------------------------
            frame = buffer.get()

            if frame is None:
                time.sleep(0.01)
                continue

            # 记录当前时间
            now = time.time()

            # ----------------------------------------------------
            # 获取前端选择类别
            # ----------------------------------------------------
            display_classes = get_display_classes()
            if display_classes is None:
                display_classes = []
            display_classes = list(display_classes)

            # ----------------------------------------------------
            # 类别变化时才更新模型
            # ----------------------------------------------------
            if display_classes != last_classes:
                print("\n" + "=" * 40)
                print(f"[WEB] 前端选择类别: {display_classes}")
                print("=" * 40)

                if display_classes:
                    try:
                        model.set_classes(display_classes)
                        last_classes = display_classes.copy()
                        print("[YOLO] set_classes 成功")
                        try:
                            print("[YOLO] model.names =", model.names)
                        except Exception:
                            pass
                    except Exception as exc:
                        print("[YOLO] set_classes 失败:", exc)
                        traceback.print_exc()
                        # 更新帧但不检测
                        update_frame(frame)
                        continue
                else:
                    last_classes = []

            # ----------------------------------------------------
            # 没有选择任何类别 → 显示原始画面
            # ----------------------------------------------------
            if not display_classes:
                update_frame(frame)
                time.sleep(0.001)
                continue

            # ----------------------------------------------------
            # 帧率控制：达到间隔才做推理
            # ----------------------------------------------------
            if now - last_infer_time < frame_interval:
                # 还没到推理时间 → 用上一次结果继续画框
                if last_vis_frame is not None:
                    update_frame(last_vis_frame)
                else:
                    update_frame(frame)
                continue

            last_infer_time = now

            # ====================================================
            # YOLO 推理
            # ====================================================
            results = model.predict(frame, conf=0.25, verbose=False)

            if not results:
                update_frame(frame)
                continue

            res = results[0]

            # ----------------------------------------------------
            # 收集检测结果
            # ----------------------------------------------------
            objects = []

            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id < 0 or cls_id >= len(display_classes):
                    print(f"[YOLO] 无效类别 ID: {cls_id}, 当前类别: {display_classes}")
                    continue

                name = display_classes[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                obj = {
                    "name": name,
                    "conf": conf,
                    "box": [x1, y1, x2, y2]
                }
                objects.append(obj)

                print(f"conf={conf:.3f}, box={[x1,y1,x2,y2]}")

            # ----------------------------------------------------
            # 安全帽异常检测
            # ----------------------------------------------------
            abnormal, person = check_no_hat(objects)
            abnormal_conf = person["conf"] if (abnormal and person) else 0

            # ----------------------------------------------------
            # 更新持续显示的变量（关键！）
            # ----------------------------------------------------
            last_objects = objects
            last_abnormal_state = abnormal

            # ----------------------------------------------------
            # 绘制画面
            # ----------------------------------------------------
            vis_frame = draw_detections(frame, objects, abnormal)
            last_vis_frame = vis_frame  # 保存供后续帧复用

            # ====================================================
            # 保存逻辑：按检测到的物体种类保存
            # ====================================================
            should_save = False
            save_abnormal = False

            # 获取当前检测到的物体种类集合
            current_classes = set(obj["name"] for obj in objects)

            # 只要检测到目标物体，按每秒一帧的频率保存
            if current_classes:
                if last_detected_classes is None:
                    # 首次检测到物体，立即保存
                    should_save = True
                    last_detected_classes = set()
                    last_object_save_time = 0  # 初始化上次保存时间
                else:
                    # 每秒保存一帧
                    if now - last_object_save_time >= 1.0:
                        should_save = True
                        last_object_save_time = now

                # 更新已检测到的物体种类
                last_detected_classes = last_detected_classes | current_classes

            # 异常情况单独处理（同样每秒保存）
            if abnormal:
                if now - last_abnormal_save_time >= 1.0:
                    save_abnormal = True
                    last_abnormal_save_time = now

            if should_save or save_abnormal:
                frame_time = datetime.datetime.now()
                filename = f"{camera_id}_{int(frame_time.timestamp()*1000)}.jpg"
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
                        "event_id": next_event_id,
                        "location": location,
                        "camera_type": camera_type,
                    })

                    put_csv_queue({
                        "id": next_event_id,
                        "event": get_event_name(objects, abnormal),
                        "time": frame_time,
                        "location": location,
                        "abnormal": abnormal,
                        "camera_type": camera_type,
                    })

                    next_event_id += 1
                else:
                    print("[IMAGE SAVE ERROR]", path)

            # ====================================================
            # 推送给网页 + 本地显示
            # ====================================================
            update_frame(vis_frame)

            # 本地弹窗（如果有 GUI 环境）
            try:
                cv2.imshow("rear RTSP", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                # headless 环境，跳过本地窗口
                pass

        except Exception as exc:
            print("\n[MAIN ERROR]", exc)
            traceback.print_exc()
            try:
                update_frame(frame)
            except Exception:
                pass
            time.sleep(0.01)

    # ============================================================
    # 退出
    # ============================================================
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
