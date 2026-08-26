import json
import time
import threading
import atexit
import requests
import traceback
import os
from aliyunsdkcore.client import AcsClient
from aliyunsdkdysmsapi.request.v20170525.SendSmsRequest import SendSmsRequest

# ✅ 调试用（验证通过后请删掉）
ACCESS_KEY_ID = "未知"
ACCESS_KEY_SECRET = "不给你看"

SIGN_NAME = "舟山市千鹏无人机科技"       # ✅ 图1里的签名
TEMPLATE_CODE = "SMS_505370126"         # ✅ 图2里的模板CODE（已修正！）
REGION = "cn-hangzhou"

class AlertCenter:
    def __init__(self, path="alert_state.json", cooldown_seconds=300):
        self.path = path
        self.cooldown = cooldown_seconds
        self.lock = threading.Lock()
        self.state = self._load()
        atexit.register(self._save)

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save(self):
        with self.lock:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False)

    def should_send_sms(self, alert_type):
        """
        alert_type: "no_hat" / "person" / "car"
        """
        with self.lock:
            now = time.time()
            last = self.state.get(alert_type, 0)

            if now - last >= self.cooldown:
                self.state[alert_type] = now
                return True

            return False

    def rollback(self, alert_type):
        with self.lock:
            now = time.time()
            last = self.state.get(alert_type, 0)
            # 只回滚 10 秒内写入的时间
            if now - last < 10:
                self.state[alert_type] = 0



def load_message_key():
    """
    从 alert_center.py 同目录下的 message_key.json
    读取阿里云 AK / SK
    """

    try:
        # 获取当前 alert_center.py 所在目录
        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        key_path = os.path.join(
            base_dir,
            "message_key.json"
        )

        with open(
            key_path,
            "r",
            encoding="utf-8"
        ) as f:
            config = json.load(f)

        access_key_id = config.get(
            "ACCESS_KEY_ID"
        )

        access_key_secret = config.get(
            "ACCESS_KEY_SECRET"
        )

        if not access_key_id:
            raise ValueError(
                "message_key.json 中没有 ACCESS_KEY_ID"
            )

        if not access_key_secret:
            raise ValueError(
                "message_key.json 中没有 ACCESS_KEY_SECRET"
            )

        return access_key_id, access_key_secret

    except FileNotFoundError:
        print(
            "[SMS KEY ERROR] "
            "找不到 message_key.json"
        )
        raise

    except json.JSONDecodeError as e:
        print(
            "[SMS KEY ERROR] "
            f"message_key.json 格式错误: {e}"
        )
        raise

    except Exception as e:
        print(
            "[SMS KEY ERROR] "
            f"读取 AK/SK 失败: {e}"
        )
        raise
def send_sms(phone, message, location="未知位置", conf=None):
    try:
        # =====================================================
        # 读取 message_key.json
        # =====================================================
        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        key_path = os.path.join(
            base_dir,
            "message_key.json"
        )

        with open(
            key_path,
            "r",
            encoding="utf-8"
        ) as f:
            key_config = json.load(f)

        access_key_id = key_config.get("ACCESS_KEY_ID")
        access_key_secret = key_config.get("ACCESS_KEY_SECRET")

        # 检查 AK/SK
        if not access_key_id:
            print("[SMS ERROR] message_key.json 中没有 ACCESS_KEY_ID")
            return False

        if not access_key_secret:
            print("[SMS ERROR] message_key.json 中没有 ACCESS_KEY_SECRET")
            return False

        print("[SMS] AK/SK 读取成功")

        client = AcsClient(
            ACCESS_KEY_ID,
            ACCESS_KEY_SECRET,
            "cn-hangzhou"
        )

        template_param = {
            "patient_name": location,
            "alert_type": message,
            "value": f"{conf:.2f}" if conf is not None else "1"
        }

        req = SendSmsRequest()
        req.set_PhoneNumbers(phone)
        req.set_SignName("舟山市千鹏无人机科技")
        req.set_TemplateCode("SMS_505370126")
        req.set_TemplateParam(json.dumps(template_param, ensure_ascii=False))

        resp = client.do_action_with_exception(req)
        print("[SMS SEND SUCCESS]", resp.decode())
        return True

    except Exception as e:
        print("[SMS SEND ERROR]", str(e))
        traceback.print_exc()
        return False