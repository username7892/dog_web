from web_server_2 import update_frame, start_web
import web_server_2

import cv2
import os
import time
import threading
import torch

from ultralytics import YOLOE


# =============================
# RTSP（子码流102）
# =============================

RTSP_URL = (
    "rtsp://100.95.170.3:8555/camera-rear"
)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
)


# =============================
# 环形缓存（单个）
# =============================

class RingBuffer:
    def __init__(self, size=30):  # 子码流帧率可能较低，缓存适当减小
        self.size = size
        self.buffer = [None] * size
        self.index = 0
        self.count = 0
        self.lock = threading.Lock()

    def put(self, frame):
        with self.lock:
            self.buffer[self.index] = frame
            self.index = (self.index + 1) % self.size
            if self.count < self.size:
                self.count += 1

    def get(self, delay):
        with self.lock:
            if self.count <= delay:
                return None
            idx = (self.index - delay - 1) % self.size
            return self.buffer[idx]


# =============================
# 双CAP管理
# =============================

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
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _read(self, cap):
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
        return None

    def _fix_cap(self, num):
        print(f"[fix] 开始修复 cap{num}")
        while self.running and self.fixing == num:
            cap = self._open()
            if cap is not None:
                if num == 1:
                    if self.cap1:
                        self.cap1.release()
                    self.cap1 = cap
                else:
                    if self.cap2:
                        self.cap2.release()
                    self.cap2 = cap
                print(f"[fix] cap{num} 修复完成，待命")
                self.fixing = None
                return
            print(f"[fix] cap{num} 修复失败，1秒后重试")
            time.sleep(1)

    def run(self):
        print("[init] 创建 cap1...")
        self.cap1 = self._open()
        time.sleep(0.3)
        print("[init] 创建 cap2...")
        self.cap2 = self._open()

        if self.cap1 is None and self.cap2 is None:
            print("[init] 两个cap都建失败，退出")
            self.running = False
            return

        if self.cap1 is not None:
            self.active = 1
        else:
            self.active = 2

        print(f"[run] 使用 cap{self.active}")

        while self.running:
            cap = self.cap1 if self.active == 1 else self.cap2

            if cap is None:
                self._switch()
                continue

            frame = self._read(cap)

            if frame is not None:
                self.buffer.put(frame)
            else:
                print(f"[warn] cap{self.active} 读不到帧，立刻切换")
                self._kill_active()
                self._switch()

    def _kill_active(self):
        if self.active == 1:
            if self.cap1:
                self.cap1.release()
            self.cap1 = None
            self.fixing = 1
            threading.Thread(target=self._fix_cap, args=(1,), daemon=True).start()
        else:
            if self.cap2:
                self.cap2.release()
            self.cap2 = None
            self.fixing = 2
            threading.Thread(target=self._fix_cap, args=(2,), daemon=True).start()

    def _switch(self):
        if self.active == 1:
            self.active = 2
            print("[switch] -> cap2")
        else:
            self.active = 1
            print("[switch] -> cap1")

    def stop(self):
        self.running = False
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()


# =============================
# 主程序
# =============================

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = YOLOE("yoloe-v8s-seg.pt")
    model.to(device)

    buffer = RingBuffer(30)

    rtsp = DualRTSP(RTSP_URL, buffer)
    threading.Thread(target=rtsp.run, daemon=True).start()

    threading.Thread(
        target=start_web,
        daemon=True
    ).start()

    print("等待缓冲区...")
    time.sleep(3)

    delay = 15  # 子码流延迟可以小一些
    result = None
    frame_id = 0

    print("开始检测（子码流），按 'q' 退出")

    last_classes = []

    while True:
        frame = buffer.get(delay)
        if frame is None:
            time.sleep(0.01)
            continue

        frame_id += 1

        try:
            now_classes = web_server_2.current_classes.copy()

            if now_classes != last_classes:
                print("YOLOE更新类别:", now_classes)
                model.set_classes(now_classes)
                last_classes = now_classes.copy()

            res = model.predict(frame, conf=0.25, verbose=False)
            result = res[0].plot()
        except Exception:
            result = frame

        # 状态显示
        c1 = "OK" if rtsp.cap1 is not None else "NONE"
        c2 = "OK" if rtsp.cap2 is not None else "NONE"
        fix = rtsp.fixing

        cv2.putText(result, f"Sub Stream | active:cap{rtsp.active}", (10, 420),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"cap1:{c1} cap2:{c2}", (10, 442),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if fix is not None:
            cv2.putText(result, f"fixing:cap{fix}", (10, 462),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow("Dual RTSP - Sub Stream", result)
        update_frame(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rtsp.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
