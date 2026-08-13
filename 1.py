import cv2

url = "rtsp://10.21.34.104:8555/camera-front"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

print("opened:", cap.isOpened())

for i in range(20):
    ret, frame = cap.read()

    if not ret:
        print("read failed")
        break

    print(
        "frame:",
        i,
        "shape:",
        frame.shape,
        "width:",
        frame.shape[1],
        "height:",
        frame.shape[0]
    )

cap.release()