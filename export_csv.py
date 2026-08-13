import pymysql
import csv
from collections import defaultdict

OUTPUT_PATH = "yolo_result.csv"

CSV_HEADERS = [
    "ID", "time", "objects",
    "location", "abnormal",
    "现场已告警", "短信通知",
    "event", "camera_type"
]

CAMERA_LOCATIONS = {
    "cam01": "一号摄像头",
    "cam02": "二号摄像头",
}
CAMERA_TYPES = {
    "cam01": "前摄",
    "cam02": "后摄",
}

CLASS_NAME_CN = {
    "person": "人",
    "helmet": "安全帽",
}

def to_cn(n):
    return CLASS_NAME_CN.get(n, n)

db = pymysql.connect(
    host="localhost",
    user="root",
    password="123456",
    database="yolo_images",
    charset="utf8mb4"
)
cur = db.cursor()

sql = """
SELECT
    i.id,
    i.frame_time,
    i.camera_id,
    d.class_name,
    COUNT(*) cnt,
    MAX(CASE WHEN ae.image_id IS NOT NULL THEN 1 ELSE 0 END) abnormal
FROM images i
LEFT JOIN detections d ON d.image_id = i.id
LEFT JOIN abnormal_events ae ON ae.image_id = i.id
GROUP BY i.id, i.frame_time, i.camera_id, d.class_name
ORDER BY i.id, d.class_name
"""

cur.execute(sql)
rows = cur.fetchall()

# 按图片聚合
data = defaultdict(list)
meta = {}

for img_id, frame_time, cam_id, cls, cnt, abnormal in rows:
    if img_id not in meta:
        meta[img_id] = {
            "time": frame_time,
            "camera_id": cam_id,
            "abnormal": bool(abnormal),
        }
    if cls:
        data[img_id].append((cls, cnt))

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(CSV_HEADERS)

    for img_id in sorted(meta):
        m = meta[img_id]
        loc = CAMERA_LOCATIONS.get(m["camera_id"], m["camera_id"])
        ctype = CAMERA_TYPES.get(m["camera_id"], "")
        tstr = m["time"].strftime("%Y-%m-%d %H:%M:%S")

        # ===== 无检测目标 =====
        if not data[img_id]:
            w.writerow([
                img_id, tstr, "未检测到目标",
                loc, "否",
                "否", "否", "", ctype
            ])
            continue

        # ===== 有检测目标：第一个写主行，其余写 1.1 / 1.2 ... =====
        objs = data[img_id]
        for idx, (cls, cnt) in enumerate(objs):
            if idx == 0:
                # 主行
                row = [
                    img_id,
                    tstr,
                    f"{cnt}{to_cn(cls)}",
                    loc,
                    "是" if m["abnormal"] else "否",
                    "否",  # 现场已告警（默认）
                    "否",  # 短信通知（默认）
                    "未佩戴安全帽" if m["abnormal"] else "",
                    ctype,
                ]
            else:
                # 子行：ID = img_id.1, img_id.2 ...
                row = [
                    f"{img_id}.{idx}",
                    "",
                    f"{cnt}{to_cn(cls)}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            w.writerow(row)

cur.close()
db.close()
print("✅ 导出完成")
