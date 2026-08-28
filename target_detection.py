class CustomSplitRule:
    SPLIT_MAP = {
        "1": ["person"],
        "2": ["helmet"],
        "person without helmet": ["person", "helmet"]
    }

    @classmethod
    def split(cls, object_name):
        """
        拆分 object
        如果不是自定义 key，则返回其本身（不做语义处理）
        """
        return cls.SPLIT_MAP.get(object_name, [object_name])

    @classmethod
    def split_all(cls, object_set):
        """
        对一组 object 进行拆分
        """
        result = set()
        for item in object_set:
            for atom in cls.split(item):
                result.add(atom)
        return result


# ============================================================
# 判断函数（放在对应类类型中）
# ============================================================

class HelmetDetector:
    """
    对应：
    person / helmet / person without helmet
    """

    @staticmethod
    def check_no_hat(objects):
        persons = [o for o in objects if o["name"] == "person"]
        helmets = [o for o in objects if o["name"] == "helmet"]

        if not persons:
            return False, None

        person_heads = []
        for p in persons:
            x1, y1, x2, y2 = p["box"]
            person_heads.append({
                "person": p,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y1 + (y2 - y1) * 0.3
            })

        matched = set()
        for h in helmets:
            hx1, hy1, hx2, hy2 = h["box"]
            cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
            best_p, best_dist = None, float("inf")

            for i, head in enumerate(person_heads):
                if head["x1"] < cx < head["x2"] and head["y1"] < cy < head["y2"]:
                    dist = abs(cy - (head["y1"] + head["y2"]) / 2)
                    if dist < best_dist:
                        best_dist, best_p = dist, i

            if best_p is not None:
                matched.add(best_p)

        for i, head in enumerate(person_heads):
            if i not in matched:
                return True, head["person"]

        return False, None