import cv2
import os
import time
import threading
import torch

from ultralytics import YOLOE


# =============================
# RTSP
# =============================

RTSP_URL = (
    "rtsp://admin:dhlb839.@192.168.50.64:554/"
    "Streaming/Channels/101"
)


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
)



# =============================
# 环形缓存
# =============================

class RingBuffer:

    def __init__(self, size=50):

        self.size = size

        self.buffer = [None] * size

        self.index = 0

        self.count = 0

        self.lock = threading.Lock()



    def put(self, frame):

        with self.lock:

            self.buffer[self.index] = frame

            self.index = (
                self.index + 1
            ) % self.size


            if self.count < self.size:

                self.count += 1




    def get(self, delay):

        with self.lock:

            if self.count <= delay:

                return None


            idx = (
                self.index
                -
                delay
                -
                1
            ) % self.size


            return self.buffer[idx]




# =============================
# 双CAP管理
# =============================

class DualRTSP:


    def __init__(self,url,buffer):

        self.url=url

        self.buffer=buffer


        self.cap1=None

        self.cap2=None


        # 当前工作cap

        self.active=1


        self.running=True


        # 两个独立恢复状态

        self.fixing={
            1:False,
            2:False
        }



    # -----------------

    def _open(self):

        cap=cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG
        )


        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )


        if cap.isOpened():

            print(
                "[RTSP] connect success"
            )

            return cap


        cap.release()

        return None



    # -----------------

    def _read(self,cap):

        ret,frame=cap.read()


        if ret and frame is not None:

            return frame


        return None




    # -----------------

    # 后台恢复

    def _fix_cap(self,num):


        if self.fixing[num]:

            return


        self.fixing[num]=True


        print(
            f"[fix] start cap{num}"
        )


        while self.running:


            newcap=self._open()


            if newcap is not None:


                if num==1:


                    if self.cap1:

                        self.cap1.release()


                    self.cap1=newcap


                else:


                    if self.cap2:

                        self.cap2.release()


                    self.cap2=newcap



                print(
                    f"[fix] cap{num} ready"
                )


                self.fixing[num]=False

                return



            print(
                f"[fix] cap{num} failed retry"
            )


            time.sleep(1)



    # -----------------

    def run(self):


        print(
            "[init] open cap1"
        )


        self.cap1=self._open()


        time.sleep(0.5)


        print(
            "[init] open cap2"
        )


        self.cap2=self._open()



        if self.cap1:

            self.active=1


        elif self.cap2:

            self.active=2


        else:

            print(
                "two cap failed"
            )

            self.running=False

            return



        print(
            f"[run] active cap{self.active}"
        )



        while self.running:


            if self.active==1:

                cap=self.cap1

            else:

                cap=self.cap2




            # 当前cap不存在

            if cap is None:


                self.switch()

                time.sleep(0.1)

                continue




            frame=self._read(cap)



            if frame is not None:


                # 写入同一个缓存

                self.buffer.put(frame)



            else:


                print(
                    f"[error] cap{self.active} lost"
                )


                bad=self.active



                # 立即释放

                self.release_cap(bad)



                # 立即切换

                self.switch()



                # 后台恢复

                threading.Thread(
                    target=self._fix_cap,
                    args=(bad,),
                    daemon=True
                ).start()



                time.sleep(0.05)



    # -----------------

    def release_cap(self,num):


        if num==1:


            if self.cap1:

                self.cap1.release()


            self.cap1=None



        else:


            if self.cap2:

                self.cap2.release()


            self.cap2=None





    # -----------------

    def switch(self):


        old=self.active



        if self.active==1:

            self.active=2


        else:

            self.active=1



        print(
            f"[switch] cap{old}->cap{self.active}"
        )




    # -----------------

    def stop(self):


        self.running=False


        if self.cap1:

            self.cap1.release()


        if self.cap2:

            self.cap2.release()





# =============================
# 主程序
# =============================

def main():


    device=(
        "cuda:0"
        if torch.cuda.is_available()
        else
        "cpu"
    )


    print(
        "device:",
        device
    )


    model=YOLOE(
        "yoloe-v8s-seg.pt"
    )


    model.to(device)


    model.set_classes(
        [
            "person",
            "car",
            "dog"
        ]
    )



    buffer=RingBuffer(50)



    rtsp=DualRTSP(
        RTSP_URL,
        buffer
    )



    threading.Thread(
        target=rtsp.run,
        daemon=True
    ).start()



    print(
        "waiting buffer"
    )


    time.sleep(3)



    delay=20



    while True:


        frame=buffer.get(delay)


        if frame is None:

            time.sleep(0.01)

            continue



        try:


            res=model.predict(
                frame,
                conf=0.25,
                verbose=False
            )


            result=res[0].plot()



        except Exception:


            result=frame




        c1="OK" if rtsp.cap1 else "NONE"

        c2="OK" if rtsp.cap2 else "NONE"



        cv2.putText(
            result,
            f"active cap{rtsp.active}",
            (10,440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )


        cv2.putText(
            result,
            f"cap1:{c1} cap2:{c2}",
            (10,465),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )



        cv2.imshow(
            "dual RTSP",
            result
        )



        if cv2.waitKey(1)&0xff==ord('q'):

            break




    rtsp.stop()

    cv2.destroyAllWindows()




if __name__=="__main__":

    main()