import json
import os
import csv
import sys
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

# ============ 修复 csv 字段大小限制 ============
def _set_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)

_set_csv_field_limit()
# ================================================


# ============ 配置 ============
SENT_FILE   = 'rice_context_sentences_compressed.tsv'
NLP_FILE    = 'NLP_Rice_GTA_Database.tsv'
OUTPUT_DIR  = 'dataset'
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# 训练集 / 验证集 各自的目标正样本比例
POS_RATIO_TRAIN = 0.7    # 训练集 正:负 = 7:3
POS_RATIO_VAL   = 0.5    # 验证集 正:负 = 1:1

N_WORKERS   = os.cpu_count() or 4
CHUNK       = 5000

INSTRUCTION = (
    "You are a biological relation extraction assistant. "
    "Given a sentence, identify all entity pairs that exhibit NLP_Cooccurrence relationship. "
    "Output a JSON list of objects with keys 'entity_1' and 'entity_2'. "
    "If no such pairs exist, output an empty list []."
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)


# ============ 1. 构建 NLP 字典 ============
def build_nlp_dict():
    nlp_map = defaultdict(list)
    seen = defaultdict(set)

    with open(NLP_FILE, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        reader.fieldnames = [c.strip() for c in reader.fieldnames]

        for row in tqdm(reader, desc="读取 NLP 库", unit="行"):
            pmid = (row.get('PMID') or '').strip()
            sid  = (row.get('Sentence_ID') or '').strip()
            e1   = (row.get('RAP_ID') or '').strip()
            e2   = (row.get('Trait_Description') or '').strip()

            if not pmid or not sid or not e1 or not e2:
                continue

            key = (pmid, sid)
            pair = (e1, e2)
            if pair in seen[key]:
                continue
            seen[key].add(pair)
            nlp_map[key].append(pair)

    print(f"NLP 字典构建完成: {len(nlp_map)} 个 (PMID, Sentence_ID) 键")
    return nlp_map


# ============ 2. 第一遍扫描：统计正负样本数 ============
def count_pos_neg(valid_keys):
    n_pos = 0
    n_neg = 0

    with open(SENT_FILE, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader, desc="扫描句子(统计)", unit="行"):
            if len(row) < 3:
                continue
            key = (row[0].strip(), row[1].strip())
            if key in valid_keys:
                n_pos += 1
            else:
                n_neg += 1

    print(f"统计完成: 正样本 {n_pos} | 负样本 {n_neg}")
    return n_pos, n_neg


# ============ 3. 计算负样本保留概率 ============
def calc_neg_keep_prob(n_pos, n_neg, pos_ratio):
    """
    目标: pos / (pos + neg_kept) = pos_ratio
    => neg_kept = pos * (1 - pos_ratio) / pos_ratio
    => keep_prob = neg_kept / n_neg
    """
    if n_pos == 0 or n_neg == 0:
        return 1.0
    target_neg = n_pos * (1 - pos_ratio) / pos_ratio
    prob = min(1.0, target_neg / n_neg)
    return prob


# ============ 4. 构造样本 ============
def _process_chunk(chunk, nlp_map):
    out = []
    for pmid, sid, sentence in chunk:
        pairs = nlp_map.get((pmid, sid), [])
        output = json.dumps(
            [{"entity_1": a, "entity_2": b} for a, b in pairs],
            ensure_ascii=False
        )
        out.append({
            "instruction": INSTRUCTION,
            "input": sentence,
            "output": output,
            "pid": pmid,
            "sentence_id": sid,
        })
    return out


def build_records(nlp_map, valid_keys, neg_keep_prob):
    """
    构造 records：
      - 正样本：全部保留
      - 负样本：按 neg_keep_prob 预采样（保留足够多，供后续 train/val 各自下采样）
    """
    records = []
    pos_chunk = []
    neg_chunk = []

    rng = random.Random(RANDOM_SEED)

    with open(SENT_FILE, 'r', encoding='utf-8', newline='') as f, \
         ThreadPoolExecutor(max_workers=N_WORKERS) as executor:

        reader = csv.reader(f, delimiter='\t')
        futures = []

        def flush_pos():
            nonlocal pos_chunk
            if pos_chunk:
                futures.append(executor.submit(_process_chunk, pos_chunk, nlp_map))
                pos_chunk = []

        def flush_neg():
            nonlocal neg_chunk
            if neg_chunk:
                for pmid, sid, sentence in neg_chunk:
                    records.append({
                        "instruction": INSTRUCTION,
                        "input": sentence,
                        "output": "[]",
                        "pid": pmid,
                        "sentence_id": sid,
                    })
                neg_chunk = []

        for row in tqdm(reader, desc="构造样本", unit="行"):
            if len(row) < 3:
                continue
            pmid = row[0].strip()
            sid  = row[1].strip()
            sentence = row[2].strip()
            key = (pmid, sid)

            if key in valid_keys:
                pos_chunk.append((pmid, sid, sentence))
                if len(pos_chunk) >= CHUNK:
                    flush_pos()
            else:
                if rng.random() < neg_keep_prob:
                    neg_chunk.append((pmid, sid, sentence))
                    if len(neg_chunk) >= CHUNK:
                        flush_neg()

        flush_pos()
        flush_neg()

        for fut in tqdm(futures, desc="收集正样本", unit="块"):
            records.extend(fut.result())

    print(f"共生成 {len(records)} 条样本")
    return records


# ============ 5. 按 PMID 分组下采样到目标比例 ============
def _downsample_by_pid(items, target_n, rng):
    """按 PMID 整组抽样，尽量保留同一 PMID 的所有句子。"""
    if target_n >= len(items):
        return items[:]

    by_pid = defaultdict(list)
    for r in items:
        by_pid[r['pid']].append(r)

    pids = list(by_pid.keys())
    rng.shuffle(pids)

    kept = []
    for pid in pids:
        group = by_pid[pid]
        if len(kept) + len(group) <= target_n:
            kept.extend(group)
        else:
            remain = target_n - len(kept)
            if remain > 0:
                kept.extend(rng.sample(group, remain))
            break
    return kept


def balance_to_ratio(records, pos_ratio, seed, desc=""):
    """
    把 records 调整到 pos_ratio : (1-pos_ratio)。
    如果某一类不足，则保留全部，比例会偏离。
    """
    rng = random.Random(seed)
    pos = [r for r in records if r['output'] != '[]']
    neg = [r for r in records if r['output'] == '[]']
    n_pos, n_neg = len(pos), len(neg)

    if n_pos == 0 or n_neg == 0:
        print(f"[警告] {desc}: pos={n_pos}, neg={n_neg}，跳过比例调整")
        return records

    cur_ratio = n_pos / (n_pos + n_neg)
    if abs(cur_ratio - pos_ratio) < 0.01:
        print(f"  {desc}: 当前比例 {cur_ratio:.2%}，已接近目标 {pos_ratio:.0%}，跳过")
        return records

    if cur_ratio > pos_ratio:
        # 正样本过多 -> 下采样正样本
        target_pos = int(round(n_neg * pos_ratio / (1 - pos_ratio)))
        keep_pos = _downsample_by_pid(pos, target_pos, rng)
        keep_neg = neg
        print(f"  {desc}: 正样本过多，下采样 {n_pos} -> {len(keep_pos)}")
    else:
        # 负样本过多 -> 下采样负样本
        target_neg = int(round(n_pos * (1 - pos_ratio) / pos_ratio))
        keep_neg = _downsample_by_pid(neg, target_neg, rng)
        keep_pos = pos
        print(f"  {desc}: 负样本过多，下采样 {n_neg} -> {len(keep_neg)}")

    out = keep_pos + keep_neg
    rng.shuffle(out)
    return out


# ============ 6. 写出文件 ============
def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(data, path, chunk=10000):
    with open(path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(0, len(data), chunk),
                      desc=f"写 {os.path.basename(path)}", unit="块", leave=False):
            block = data[i:i+chunk]
            f.write('\n'.join(json.dumps(x, ensure_ascii=False) for x in block) + '\n')


# ============ 7. 主流程 ============
def main():
    # 1) 建 NLP 字典
    nlp_map = build_nlp_dict()
    valid_keys = set(nlp_map.keys())

    # 2) 第一遍扫描：统计正负样本数
    n_pos, n_neg = count_pos_neg(valid_keys)

    # 3) 负样本预采样概率：按"较宽松"的目标保留（取 train/val 中所需负样本较多的那个）
    #    train 需要 neg = pos * (1-0.7)/0.7 ≈ 0.4286 * pos
    #    val   需要 neg = pos * (1-0.5)/0.5 = 1.0 * pos
    #    所以按 val 的 1:1 来预保留，确保 val 有足够负样本
    prob_train = calc_neg_keep_prob(n_pos, n_neg, POS_RATIO_TRAIN)
    prob_val   = calc_neg_keep_prob(n_pos, n_neg, POS_RATIO_VAL)
    neg_keep_prob = max(prob_train, prob_val)
    print(f"训练集目标比例 {POS_RATIO_TRAIN:.0%} -> 负样本保留概率 {prob_train:.4f}")
    print(f"验证集目标比例 {POS_RATIO_VAL:.0%} -> 负样本保留概率 {prob_val:.4f}")
    print(f"实际采用（取较大者）: {neg_keep_prob:.4f}")

    # 4) 构造样本（正样本全保留，负样本按预采样概率保留）
    records = build_records(nlp_map, valid_keys, neg_keep_prob)

    n_pos_final = sum(1 for r in records if r['output'] != '[]')
    n_neg_final = len(records) - n_pos_final
    print(f"预采样后: {len(records)} 条 | 正 {n_pos_final} | 负 {n_neg_final} "
          f"| 正样本占比 {n_pos_final/len(records):.2%}")

    # 5) 按 PMID 分组划分 train/val（保持同一 PMID 不跨 split）
    groups_pid = [r['pid'] for r in records]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss.split(records, groups=groups_pid))

    train_records = [records[i] for i in train_idx]
    val_records   = [records[i] for i in val_idx]

    # 6) 各自下采样到目标比例
    print("调整训练集比例到 7:3 ...")
    train_records = balance_to_ratio(train_records, POS_RATIO_TRAIN,
                                     seed=RANDOM_SEED, desc="训练集")
    print("调整验证集比例到 1:1 ...")
    val_records   = balance_to_ratio(val_records, POS_RATIO_VAL,
                                     seed=RANDOM_SEED + 1, desc="验证集")

    def stat(data, name):
        pos = sum(1 for r in data if r['output'] != '[]')
        neg = len(data) - pos
        ratio = pos / len(data) if data else 0
        print(f"{name}: 共 {len(data)} 条 | 正 {pos} | 负 {neg} | 正样本占比 {ratio:.2%}")

    stat(train_records, "训练集")
    stat(val_records,   "验证集")

    # 7) 写出
    save_jsonl(train_records, os.path.join(OUTPUT_DIR, 'train.jsonl'))
    save_jsonl(val_records,   os.path.join(OUTPUT_DIR, 'val.jsonl'))
    save_json(train_records,  os.path.join(OUTPUT_DIR, 'train.json'))
    save_json(val_records,    os.path.join(OUTPUT_DIR, 'val.json'))

    print(f"已保存到 {OUTPUT_DIR}/ : train.json / val.json / train.jsonl / val.jsonl")


if __name__ == '__main__':
    main()