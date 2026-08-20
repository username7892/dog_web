运行前端网页
cd /home/user/桌面/1/m20-inspection-demo-master
./mvnw spring-boot:run -pl inspection-app

前端网页地址为localhost:8080，如果进不去就在终端运行ikuuuvpn启动一下加速器

分别在两个终端上运行前后摄像头py文件
cd /home/user/dog_web/python_yoloe

进入环境
conda activate dog_det
运行py文件
python yolo_pengRevised.py
python yolo_pengRevised_2.py

运行回放py文件
另开终端运行
python playback_web.py

