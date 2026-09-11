"""把 YOLO 检测结果推送到 GOS 巡检应用。

GOS 收到上报时会立刻采样一次雷达位姿，把像素框投影成世界坐标，
所以这里只需要上报原始像素框和画面尺寸，不需要自己算坐标。

环境变量：
    GOS_URL              巡检应用地址，默认 http://127.0.0.1:8080
    GOS_CAMERA_ID        相机标识，需与 detection.camera-yaw-offset-deg 的键对应，默认 front
    GOS_PUSH_INTERVAL    最小上报间隔（秒），默认 0.5
    GOS_SNAPSHOT_INTERVAL  告警截图最小间隔（秒），默认 30
"""

import base64
import datetime
import os
import queue
import threading
import time

import requests


class GosDetectionReporter:
    """后台线程上报检测结果，主循环不会被网络阻塞。"""

    def __init__(
        self,
        base_url=None,
        camera_id=None,
        interval=None,
        timeout=1.0,
        enabled=True,
        snapshot_interval=None,
    ):
        self.endpoint = (base_url or os.getenv("GOS_URL", "http://127.0.0.1:8080")).rstrip("/") \
            + "/api/detections"
        self.camera_id = camera_id or os.getenv("GOS_CAMERA_ID", "front")
        self.interval = float(interval if interval is not None
                              else os.getenv("GOS_PUSH_INTERVAL", "0.5"))
        self.snapshot_interval = float(
            snapshot_interval if snapshot_interval is not None
            else os.getenv("GOS_SNAPSHOT_INTERVAL", "30")
        )
        self.timeout = timeout
        self.enabled = enabled
        self._queue = queue.Queue(maxsize=1)
        self._last_sent = 0.0
        self._last_snapshot = 0.0
        self._failures = 0
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def snapshot_due(self):
        """本帧是否该带一张现场截图。编码 JPEG 有开销，先用它判断再决定要不要编。"""
        return time.time() - self._last_snapshot >= self.snapshot_interval

    def publish(
        self,
        objects,
        image_width,
        image_height=None,
        frame_time=None,
        snapshot=None,
    ):
        """提交一帧结果。objects 元素需含 name / conf / box=[x1,y1,x2,y2]。

        snapshot 传 JPEG 字节（仅告警时传），受 snapshot_interval 节流。
        """
        if not self.enabled or not objects or not image_width:
            return
        use_snapshot = False
        if snapshot and self.snapshot_due():
            use_snapshot = True
            self._last_snapshot = time.time()
        payload = {
            "camera": self.camera_id,
            "imageWidth": int(image_width),
            "imageHeight": int(image_height or 0),
            "observedAt": (frame_time or datetime.datetime.now()).isoformat(),
            "snapshotBase64": base64.b64encode(snapshot).decode("ascii")
            if use_snapshot else None,
            "objects": [
                {
                    "name": obj["name"],
                    "confidence": float(obj.get("conf", 0.0)),
                    "x1": float(obj["box"][0]),
                    "y1": float(obj["box"][1]),
                    "x2": float(obj["box"][2]),
                    "y2": float(obj["box"][3]),
                }
                for obj in objects
                if obj.get("box") and obj.get("name")
            ],
        }
        if not payload["objects"]:
            return
        # 只保留最新一帧，处理不过来时丢弃旧的。
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    def _run(self):
        while True:
            payload = self._queue.get()
            wait = self.interval - (time.time() - self._last_sent)
            if wait > 0:
                time.sleep(wait)
            try:
                response = requests.post(
                    self.endpoint, json=payload, timeout=self.timeout
                )
                self._last_sent = time.time()
                if response.status_code == 503:
                    # 机器人未定位，属预期状态，只在次数变化时提示一次。
                    self._failures += 1
                    if self._failures == 1 or self._failures % 100 == 0:
                        print(f"[GOS] 暂不接收：{response.text}")
                    continue
                if not response.ok:
                    self._failures += 1
                    print(f"[GOS] 上报失败 {response.status_code}: {response.text[:200]}")
                    continue
                if self._failures:
                    print("[GOS] 上报已恢复")
                self._failures = 0
            except Exception as error:
                self._failures += 1
                if self._failures == 1 or self._failures % 100 == 0:
                    print(f"[GOS] 上报异常: {error}")
                self._last_sent = time.time()


if __name__ == "__main__":
    # 自检：GOS 上跑着巡检应用时，POST 一个位于画面正前方的对象。
    reporter = GosDetectionReporter()
    reporter.publish(
        [{"name": "person", "conf": 0.9, "box": [900, 300, 1020, 1000]}],
        1920,
        1080,
    )
    print(f"已提交到 {reporter.endpoint}，等待上报线程发送…")
    time.sleep(3)
