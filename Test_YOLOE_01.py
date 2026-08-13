import cv2
import os
import time
import threading
import numpy as np
import torch

from ultralytics import YOLOE


# ==============================
# RTSP地址
# ==============================

RTSP_URL = (
    "rtsp://admin:dhlb839.@192.168.50.64:554/"
    "Streaming/Channels/101"
)


# ==============================
# FFmpeg参数
# ==============================

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
)


# ==============================
# 环形缓冲区
# ==============================

class RingBuffer:
    def __init__(self, size=50):
        self.size = size
        self.buffer = [None] * size
        self.write_index = 0
        self.count = 0
        self.lock = threading.Lock()

    def put(self, frame):
        with self.lock:
            self.buffer[self.write_index] = frame
            self.write_index = (
                self.write_index + 1
            ) % self.size

            if self.count < self.size:
                self.count += 1

    def get_delay_frame(self, delay):
        """
        delay:
        延迟多少帧

        delay=0 最新
        delay=20 读取20帧以前
        """
        with self.lock:
            if self.count <= delay:
                return None

            index = (
                self.write_index
                - delay
                - 1
            ) % self.size

            return self.buffer[index]

    def length(self):
        with self.lock:
            return self.count


# ==============================
# RTSP采集线程（优化重连版）
# ==============================

class RTSPReader:
    def __init__(self, url, buffer):
        self.url = url
        self.buffer = buffer
        self.cap = None
        self.running = True
        self.connected = False
        self.fail_count = 0
        self.reconnect_count = 0
        self.last_reconnect_time = 0

    def connect(self):
        print("尝试连接RTSP...")
        
        self.cap = cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG
        )
        
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )
        
        if self.cap.isOpened():
            print("RTSP连接成功")
            self.connected = True
            self.fail_count = 0
            self.reconnect_count = 0
            return True
        else:
            print("RTSP连接失败")
            self.connected = False
            return False

    def reconnect(self):
        """快速重连"""
        if self.cap:
            self.cap.release()
        
        self.connected = False
        self.reconnect_count += 1
        
        # 动态重连延迟：第一次0.1秒，之后逐渐增加
        delay = min(0.1 * self.reconnect_count, 1.0)  # 最大1秒
        time.sleep(delay)
        
        self.connect()

    def run(self):
        while self.running:
            if not self.connected:
                self.connect()
                if not self.connected:
                    time.sleep(0.1)  # 快速重试
                    continue

            ret, frame = self.cap.read()

            if ret:
                self.fail_count = 0
                self.buffer.put(frame)
            else:
                self.fail_count += 1
                
                # 只在第一次失败时打印
                if self.fail_count == 1:
                    print("RTSP读取失败，尝试重连...")
                
                # 立即重连，不等待
                self.reconnect()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ==============================
# 主程序
# ==============================

def main():
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("运行设备:", device)

    print("加载YOLOE...")
    model = YOLOE("yoloe-v8s-seg.pt")
    model.to(device)
    model.set_classes([
        "person",
        "car",
        "dog",
        "bottle",
        "cup"
    ])
    print("YOLOE加载完成")

    # 创建缓冲区
    buffer = RingBuffer(size=50)

    # RTSP线程
    reader = RTSPReader(RTSP_URL, buffer)
    t = threading.Thread(target=reader.run, daemon=True)
    t.start()

    # 延迟读取20帧
    DELAY_FRAME = 20

    last_frame_id = None
    frame_id = 0
    display_frame = None
    last_time = time.time()

    print("开始检测，按 'q' 退出")

    while True:
        frame = buffer.get_delay_frame(DELAY_FRAME)

        if frame is None:
            cv2.waitKey(10)
            continue

        frame_id += 1

        # 判断是否新帧
        if frame_id != last_frame_id:
            try:
                results = model.predict(
                    source=frame,
                    conf=0.25,
                    verbose=False
                )

                if len(results) > 0:
                    display_frame = results[0].plot()

                last_frame_id = frame_id

            except Exception as e:
                print("YOLO错误:", e)
                display_frame = frame

        if display_frame is None:
            display_frame = frame

        fps = 1 / (time.time() - last_time)
        last_time = time.time()

        show = display_frame.copy()
        cv2.putText(
            show,
            f"FPS:{fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.putText(
            show,
            f"Buffer:{buffer.length()}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.putText(
            show,
            f"Reconnects:{reader.reconnect_count}",
            (10, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1
        )

        cv2.imshow("YOLOE RTSP", show)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    reader.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()