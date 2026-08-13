

import time
import clip
import os
from ultralytics import YOLOE
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
)
import cv2

def run_optimized_yoloe_stream(source=0):
    """
    高性能 YOLOE 实时视频流检测脚本
    :param source: 视频源，0为本地摄像头，也可填 RTSP 地址
    """
    # 1. 加载模型 (根据显存选择 s/m/l，s最快，l最准)
    print("正在加载 YOLOE 模型...")

    try:
        model = YOLOE("yoloe-v8s-seg.pt")
    except Exception as e:
        print(f"模型加载失败，请检查网络或模型路径: {e}")
        return
    print("模型加载成功，开始视频流检测...")
    # 2. 打开视频源
    # cap = cv2.VideoCapture(source)
    cap = cv2.VideoCapture(
    source,
    cv2.CAP_FFMPEG
    )  
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise Exception(f"无法连接视频源: {source}")

    # 定义检测类别：人、车、狗
    prompt_classes = ["person", "car", "dog", "cup", "bottle", "chair", "sofa", "pottedplant", "tvmonitor"]
    print(f"检测类别: {prompt_classes}")
    model.set_classes(prompt_classes)
    print(2222222)
    # 优化参数：跳帧策略
    skip_frames = 30  # 每3帧检测一次，提升FPS
    frame_count = 0
    last_annotated_frame = None

    print(f"启动检测，目标: {prompt_classes} | 按 'q' 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # print("视频流结束或读取失败")
            # break
            print("RTSP断开，尝试重连")
            cap.release()
            time.sleep(2)
            cap=cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            continue
        
        frame_count += 1
        start_time = time.time()
        print(11111111111)
        # 策略：每隔 skip_frames 帧进行一次完整推理
        if frame_count % (skip_frames + 1) == 0:
            try:
                results = model.predict(
                    source=frame,
                    conf=0.25,
                    verbose=False,
                    stream=False
                )
                # 绘制当前帧结果
                if results and len(results) > 0:
                    # last_annotated_frame = results.plot()
                    last_annotated_frame = results[0].plot()
                else:
                    last_annotated_frame = frame
                # if results:
                #     last_annotated_frame = results[0].plot()
                # else:
                #     last_annotated_frame = frame
            except Exception as e:
                print(f"推理错误: {e}")
                last_annotated_frame = frame
        else:
            # 非检测帧，沿用上一帧的绘制结果，避免画面闪烁或黑屏
            if last_annotated_frame is None:
                last_annotated_frame = frame

        # 计算并显示 FPS
        fps = 1 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        cv2.putText(last_annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示画面
        cv2.imshow("YOLOE Real-time Detection", last_annotated_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    # 默认使用摄像头0，如需RTSP请修改为: run_optimized_yoloe_stream("rtsp://admin:pass@ip/stream")
    run_optimized_yoloe_stream(source="rtsp://admin:dhlb839.@192.168.50.64:554/Streaming/Channels/102")
