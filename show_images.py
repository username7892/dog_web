import pymysql
import cv2
import time
import numpy as np

# =====================
# 配置
# =====================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "yolo_images",
    "charset": "utf8mb4"
}

CAMERA_ID = "cam01"
START_TIME = "2026-08-11 16:40:49"
END_TIME   = "2026-08-11 16:41:13"

FPS = 10
INTERVAL = 1.0 / FPS

# ✅ 检索物过滤（空列表 = 不过滤，显示全部）
FILTER_CLASSES = []   # 例如: ["person", "helmet"]

# =====================
# 工具函数
# =====================

def get_db():
    return pymysql.connect(**DB_CONFIG)

def load_image(img_bytes):
    if img_bytes is None:
        return None
    if isinstance(img_bytes, str):
        return cv2.imread(img_bytes)
    if img_bytes.startswith(b'/') or img_bytes.startswith(b'images'):
        return cv2.imread(img_bytes.decode(errors='ignore'))
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def resize_keep_ratio(img, max_h=360):
    h, w = img.shape[:2]
    if h <= max_h:
        return img
    scale = max_h / h
    return cv2.resize(img, (int(w * scale), max_h))

# =====================
# 主程序
# =====================

db = get_db()
cursor = db.cursor()

# ---------- 1. 查帧 ----------
cursor.execute("""
SELECT id, image, frame_time
FROM images
WHERE camera_id = %s
  AND frame_time BETWEEN %s AND %s
ORDER BY frame_time ASC
""", (CAMERA_ID, START_TIME, END_TIME))

rows = cursor.fetchall()

if not rows:
    print("❌ 没有查到任何帧")
    exit(1)

image_ids = [r[0] for r in rows]

# ---------- 2. 查 detections ----------
cursor.execute("""
SELECT image_id, class_name, confidence
FROM detections
WHERE image_id IN ({})
""".format(','.join(['%s'] * len(image_ids))), image_ids)

dets_raw = cursor.fetchall()

# 按 image_id 分组
dets_map = {}
for img_id, cls, conf in dets_raw:
    dets_map.setdefault(img_id, []).append((cls, float(conf)))

# ---------- 3. 播放 ----------
print(f"✅ 共 {len(rows)} 帧，按 q 退出")

for img_id, img_bytes, frame_time in rows:

    img = load_image(img_bytes)
    if img is None:
        print("无法加载:", frame_time)
        continue

    # 等比缩小
    img_small = resize_keep_ratio(img, 360)

    # 时间水印
    time_str = frame_time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        img_small, time_str,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 255), 2
    )

    # ---------- 右侧文字（终端） ----------
    dets = dets_map.get(img_id, [])
    if FILTER_CLASSES:
        dets = [d for d in dets if d[0] in FILTER_CLASSES]

    print(f"\n📸 {time_str}")
    if dets:
        for cls, conf in dets:
            print(f"   ✅ {cls:<12} {conf*100:.1f}%")
    else:
        print("   ❌ 无检测")

    # ---------- 显示 ----------
    cv2.imshow("Playback", img_small)

    if cv2.waitKey(int(INTERVAL * 1000)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cursor.close()
db.close()
print("✅ 播放结束")
