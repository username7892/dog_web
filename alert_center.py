import json
import time
import threading
import atexit
import traceback
import os

from aliyunsdkcore.client import AcsClient
from aliyunsdkdysmsapi.request.v20170525.SendSmsRequest import SendSmsRequest


SIGN_NAME = "舟山市千鹏无人机科技"
TEMPLATE_CODE = "SMS_505370126"
REGION = "cn-hangzhou"


class AlertCenter:

    def __init__(self, path="alert_state.json", cooldown_seconds=300):
        self.path = os.path.abspath(path)
        self.cooldown = cooldown_seconds
        self.lock = threading.Lock()

        self._ensure_file()
        atexit.register(self._save)

    def _ensure_file(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False)

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)

        except FileNotFoundError:
            return {}

        except json.JSONDecodeError:
            print("[ALERT STATE ERROR] alert_state.json 格式错误")
            return {}

    def _save(self, state=None):
        if state is None:
            state = self._load()

        temp_path = self.path + ".tmp"

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

        os.replace(temp_path, self.path)

    def should_send_sms(self, alert_type):
        with self.lock:
            state = self._load()
            now = time.time()
            last = state.get(alert_type, 0)

            if now - last >= self.cooldown:
                state[alert_type] = now
                self._save(state)

                print(f"[SMS] 允许发送: {alert_type}")
                return True

            remaining = self.cooldown - (now - last)
            print(
                f"[SMS] {alert_type} 仍在冷却中，"
                f"剩余 {remaining:.1f} 秒"
            )
            return False

    def rollback(self, alert_type):
        with self.lock:
            state = self._load()
            now = time.time()
            last = state.get(alert_type, 0)

            if now - last < 10:
                state[alert_type] = 0
                self._save(state)
                print(f"[SMS] 回滚成功: {alert_type}")


def load_message_key():
    try:
        key_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "message_key.json"
        )

        with open(key_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        access_key_id = config.get("ACCESS_KEY_ID")
        access_key_secret = config.get("ACCESS_KEY_SECRET")

        if not access_key_id:
            raise ValueError("message_key.json 中没有 ACCESS_KEY_ID")

        if not access_key_secret:
            raise ValueError("message_key.json 中没有 ACCESS_KEY_SECRET")

        return access_key_id, access_key_secret

    except FileNotFoundError:
        print("[SMS KEY ERROR] 找不到 message_key.json")
        raise

    except json.JSONDecodeError as e:
        print(f"[SMS KEY ERROR] message_key.json 格式错误: {e}")
        raise

    except Exception as e:
        print(f"[SMS KEY ERROR] 读取 AK/SK 失败: {e}")
        raise


def send_sms(phone, message, location="未知位置", conf=None):
    try:
        access_key_id, access_key_secret = load_message_key()

        client = AcsClient(
            access_key_id,
            access_key_secret,
            REGION
        )

        template_param = {
            "patient_name": location,
            "alert_type": message,
            "value": f"{conf:.2f}" if conf is not None else "1"
        }

        req = SendSmsRequest()
        req.set_PhoneNumbers(phone)
        req.set_SignName(SIGN_NAME)
        req.set_TemplateCode(TEMPLATE_CODE)
        req.set_TemplateParam(
            json.dumps(template_param, ensure_ascii=False)
        )

        resp = client.do_action_with_exception(req)

        print("[SMS RESPONSE]", resp.decode("utf-8"))
        return True

    except Exception as e:
        print("[SMS SEND ERROR]", repr(e))
        traceback.print_exc()
        return False