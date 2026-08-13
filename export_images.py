import pymysql
import os


SAVE_DIR = "./export_jpg"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# 连接数据库
db = pymysql.connect(
    host="localhost",
    user="root",
    password="123456",
    database="yolo_images",
    charset="utf8mb4"
)


cursor = db.cursor()



# 查询图片 + 检测物体
sql = """
SELECT

images.id,

images.image,

images.frame_time,

GROUP_CONCAT(
detections.class_name
)

FROM images


LEFT JOIN detections

ON images.id = detections.image_id


GROUP BY images.id


ORDER BY images.id

"""


cursor.execute(sql)


rows = cursor.fetchall()


print("图片数量:", len(rows))



for row in rows:


    image_id = row[0]

    image_bytes = row[1]

    frame_time = row[2]

    class_names = row[3]



    # 没有检测结果
    if class_names is None:

        class_names = "unknown"



    # 多个物体去重

    class_names = "_".join(
        sorted(
            set(
                class_names.split(",")
            )
        )
    )



    # 时间格式

    time_str = frame_time.strftime(
        "%Y%m%d_%H%M%S"
    )



    filename = (

        f"{SAVE_DIR}/"

        f"{class_names}_"

        f"{time_str}.jpg"

    )



    with open(
        filename,
        "wb"
    ) as f:

        f.write(image_bytes)



    print(
        "导出:",
        filename
    )



cursor.close()

db.close()


print("完成")
