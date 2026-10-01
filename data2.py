# scripts/prepare_data.py
import json
import re
import random
from collections import Counter

RAW_PATH = "dataset/train_pair.jsonl"
OUT_PATH = "dataset/train_sft.jsonl"
STATS_PATH = "dataset/stats.json"

INSTRUCTION = (
    "判断下列水稻文本中，实体E1与候选触发词E2之间是否存在关系。"
    "存在输出1，不存在输出0。"
)

def normalize_text(text: str) -> str:
    """统一空格与标签格式"""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 统一 E1 标签：<E1> xxx </E1> -> <E1>xxx</E1>
    text = re.sub(r"<E1>\s*(.*?)\s*</E1>", r"<E1>\1</E1>", text)
    # 统一 E2 标签：[E2] xxx [/E2] -> [E2]xxx[/E2]
    text = re.sub(r"\[E2\]\s*(.*?)\s*\[/E2\]", r"[E2]\1[/E2]", text)
    return text

def is_valid(item: dict) -> bool:
    text = item.get("text", "")
    if "<E1>" not in text or "</E1>" not in text:
        return False
    if "[E2]" not in text or "[/E2]" not in text:
        return False
    if item.get("label") not in (0, 1):
        return False
    return True

def main():
    seen = set()
    data = []
    stats = Counter()

    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not is_valid(item):
                stats["invalid"] += 1
                continue

            text = normalize_text(item["text"])
            label = int(item["label"])

            key = (text, label)
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)

            data.append({"text": text, "label": label})
            stats[f"label_{label}"] += 1

    random.shuffle(data)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for item in data:
            record = {
                "instruction": INSTRUCTION,
                "input": item["text"],
                "output": str(item["label"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats["total"] = len(data)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("数据统计：", dict(stats))
    print(f"已写出 {len(data)} 条到 {OUT_PATH}")

if __name__ == "__main__":
    main()