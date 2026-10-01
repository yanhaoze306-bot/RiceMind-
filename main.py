# scripts/train_full.py
import os
import json
import random
import torch
from collections import Counter
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer

# ==================== 配置 ====================
MODEL_PATH = r"D:\AI_LLM\llama"
DATA_PATH  = r"/dataset/train_sft.jsonl"
OUTPUT_DIR = r"/outputs/rice_re_lora"

MAX_LEN = 256
INSTRUCTION = (
    "判断下列水稻文本中，实体E1与候选触发词E2之间是否存在关系。"
    "存在输出1，不存在输出0。"
)

TRAIN_POS = 21000
TRAIN_NEG = 9000
VAL_POS   = 1000
VAL_NEG   = 1000

SEED = 42
random.seed(SEED)

# ==================== 1. 读取 + 按比例采样 ====================
print("=" * 60)
print("[1/5] 读取原始数据并采样 ...")

pos, neg = [], []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if int(item["output"]) == 1:
            pos.append(item)
        else:
            neg.append(item)

print(f"原始正样本：{len(pos)}")
print(f"原始负样本：{len(neg)}")

need_pos = TRAIN_POS + VAL_POS
need_neg = TRAIN_NEG + VAL_NEG
assert len(pos) >= need_pos, f"正样本不足：需要 {need_pos}，只有 {len(pos)}"
assert len(neg) >= need_neg, f"负样本不足：需要 {need_neg}，只有 {len(neg)}"

random.shuffle(pos)
random.shuffle(neg)

val_raw   = pos[:VAL_POS] + neg[:VAL_NEG]
train_raw = pos[VAL_POS:VAL_POS + TRAIN_POS] + neg[VAL_NEG:VAL_NEG + TRAIN_NEG]

random.shuffle(val_raw)
random.shuffle(train_raw)

print(f"训练集：{len(train_raw)} 条，正负：{Counter(int(x['output']) for x in train_raw)}")
print(f"验证集：{len(val_raw)} 条，正负：{Counter(int(x['output']) for x in val_raw)}")

# ==================== 2. 构造 Dataset ====================
print("=" * 60)
print("[2/5] 构造 Dataset ...")

def format_sample(item):
    prompt = f"{INSTRUCTION}\n\n{item['input']}"
    return {"prompt": prompt, "response": str(item["output"])}

train_ds = Dataset.from_list([format_sample(x) for x in train_raw])
val_ds   = Dataset.from_list([format_sample(x) for x in val_raw])

# ==================== 3. 加载模型 + LoRA ====================
print("=" * 60)
print("[3/5] 加载模型和 tokenizer ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
)
model.config.use_cache = False

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# ==================== 4. Tokenize ====================
print("=" * 60)
print("[4/5] Tokenize 数据 ...")

def tokenize_fn(example):
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(example["response"], add_special_tokens=False)["input_ids"]
    response_ids = response_ids + [tokenizer.eos_token_id]

    # ===== 关键：从 prompt 头部截断，保证 response 完整保留 =====
    total = len(prompt_ids) + len(response_ids)
    if total > MAX_LEN:
        keep = MAX_LEN - len(response_ids)
        if keep <= 0:
            input_ids = response_ids[:MAX_LEN]
            labels = input_ids.copy()
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            }
        prompt_ids = prompt_ids[-keep:]
    # =========================================================

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }

train_tokenized = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
val_tokenized   = val_ds.map(tokenize_fn, remove_columns=val_ds.column_names)

# ===== 验证：统计有多少样本的 labels 全 -100 =====
def count_all_ignore(ds, name):
    n_total = len(ds)
    n_all_ignore = 0
    n_truncated = 0
    for ex in ds:
        valid = sum(1 for l in ex["labels"] if l != -100)
        if valid == 0:
            n_all_ignore += 1
        if len(ex["input_ids"]) == MAX_LEN:
            n_truncated += 1
    print(f"[{name}] 总 {n_total}，labels 全 -100 的 {n_all_ignore}，长度达 MAX_LEN 的 {n_truncated}")

count_all_ignore(train_tokenized, "训练集")
count_all_ignore(val_tokenized, "验证集")

# ==================== 5. 训练 ====================
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer

pos_token_id = tokenizer.encode("1", add_special_tokens=False)[0]
neg_token_id = tokenizer.encode("0", add_special_tokens=False)[0]
print(f"正类 token id：{pos_token_id}，负类 token id：{neg_token_id}")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # 移位对齐
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    valid = np.isin(labels, [pos_token_id, neg_token_id])
    preds = preds[valid]
    labels = labels[valid]

    if len(labels) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_true = (labels == pos_token_id).astype(int)
    y_pred = (preds == pos_token_id).astype(int)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


print("=" * 60)
print("[5/5] 开始训练 ...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    disable_tqdm=False,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    ),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("=" * 60)
print(f"训练完成，模型保存到 {OUTPUT_DIR}")