import json, os, random, re

random.seed(42)

E1_OPEN, E1_CLOSE = "<E1>", "</E1>"
E2_OPEN, E2_CLOSE = "[E2]", "[/E2]"

TRAIN_IN  = r"D:\pythonProject\dataset\train.jsonl"
VAL_IN    = r"D:\pythonProject\dataset\val.jsonl"
TRAIN_OUT = r"D:\pythonProject\dataset\train_pair.jsonl"
VAL_OUT   = r"D:\pythonProject\dataset\val_pair.jsonl"

NEG_PER_POS = 2              # 每个正例配 2 个随机负例
MAX_PAIRS_PER_SENT = 50      # 单句最多抽这么多正例
MIN_SPAN = 1                 # 随机片段最短词数
MAX_SPAN = 3                 # 随机片段最长词数
MAX_TRIES = 20               # 每个负例最多尝试几次避免重复


def tokenize_words_with_pos(sentence):
    """返回 [(word, start, end), ...]，按空白+标点切分，保留位置"""
    spans = []
    for m in re.finditer(r"[A-Za-z0-9\-]+", sentence):
        spans.append((m.group(0), m.start(), m.end()))
    return spans


def random_span(sentence, word_spans):
    """随机抽 1-3 个连续词作为伪实体，返回 (start, end, text)"""
    if not word_spans:
        return None
    n = random.randint(MIN_SPAN, min(MAX_SPAN, len(word_spans)))
    i = random.randint(0, len(word_spans) - n)
    start = word_spans[i][1]
    end = word_spans[i + n - 1][2]
    return start, end, sentence[start:end]


def mark_e2(sentence, start, end):
    return sentence[:start] + E2_OPEN + sentence[start:end] + E2_CLOSE + sentence[end:]


def convert(in_path, out_path):
    n_pos = n_neg = n_skip = 0
    with open(in_path, "r", encoding="utf-8") as f, \
         open(out_path, "w", encoding="utf-8") as fo:
        for line in f:
            try:
                obj = json.loads(line)
                sentence = obj["input"].strip()
                pairs = json.loads(obj["output"])
            except Exception:
                continue

            if not pairs:
                continue

            word_spans = tokenize_words_with_pos(sentence)
            if not word_spans:
                continue

            # ---------- 正例 ----------
            pos_set = set()
            for pr in pairs[:MAX_PAIRS_PER_SENT]:
                e1 = pr["entity_1"].strip()
                e2 = pr["entity_2"].strip()
                if not e1 or not e2:
                    continue
                low = sentence.lower()
                i = low.find(e2.lower())
                if i == -1:
                    n_skip += 1
                    continue
                marked = mark_e2(sentence, i, i + len(e2))
                text = f"{E1_OPEN} {e1} {E1_CLOSE} {marked}"
                pos_set.add((e1, e2.lower()))
                fo.write(json.dumps({"text": text, "label": 1},
                                    ensure_ascii=False) + "\n")
                n_pos += 1

            if not pos_set:
                continue

            # ---------- 负例（纯随机）----------
            e1_list = list({pr["entity_1"] for pr in pairs})
            n_target = NEG_PER_POS * len(pos_set)
            made = 0
            tries = 0
            while made < n_target and tries < n_target * MAX_TRIES:
                tries += 1
                e1 = random.choice(e1_list)
                span = random_span(sentence, word_spans)
                if span is None:
                    break
                s, e, txt = span
                # 剔除和任何正例 entity_2 重叠的片段
                if any(txt.lower() == e2 for _, e2 in pos_set):
                    continue
                marked = mark_e2(sentence, s, e)
                text = f"{E1_OPEN} {e1} {E1_CLOSE} {marked}"
                fo.write(json.dumps({"text": text, "label": 0},
                                    ensure_ascii=False) + "\n")
                made += 1
                n_neg += 1

    print(f"{in_path} -> {out_path}: 正={n_pos}, 负={n_neg}, 跳过(e2不在句)={n_skip}")


if __name__ == "__main__":
    convert(TRAIN_IN, TRAIN_OUT)
    convert(VAL_IN,   VAL_OUT)