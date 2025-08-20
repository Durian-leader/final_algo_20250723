import os
import json
import random
import torchaudio
from collections import defaultdict

def collect_durations(root_dir):
    label_files, label_durations = defaultdict(list), defaultdict(float)
    for label in os.listdir(root_dir):
        ldir = os.path.join(root_dir, label)
        if not os.path.isdir(ldir): continue
        for fname in os.listdir(ldir):
            if fname.endswith(".wav"):
                path = os.path.join(ldir, fname)
                try:
                    info = torchaudio.info(path)
                    dur = info.num_frames / info.sample_rate
                    label_files[label].append({"path": path, "duration": dur, "label": label})
                    label_durations[label] += dur
                except:
                    continue
    return label_files, label_durations

def compute_target_durations(label_durations, label_weights):
    limiting = min(label_weights, key=lambda l: label_durations[l] / label_weights[l])
    base = label_durations[limiting] #/ label_weights[limiting]
    return {label: base * w for label, w in label_weights.items()}, limiting, base

def random_select(files, target):
    random.shuffle(files)
    selected, dur = [], 0
    for f in files:
        if dur + f["duration"] <= target:
            selected.append(f)
            dur += f["duration"]
        if dur >= target:
            break
    return selected, dur

def split_train_val_test(files, ratio=(0.8, 0.1, 0.1)):
    random.shuffle(files)
    total = sum(f["duration"] for f in files)
    t_target, v_target = total * ratio[0], total * ratio[1]
    train, val, test = [], [], []
    dt = dv = 0
    for f in files:
        if dt + f["duration"] <= t_target:
            train.append(f); dt += f["duration"]
        elif dv + f["duration"] <= v_target:
            val.append(f); dv += f["duration"]
        else:
            test.append(f)
    return train, val, test

def write_split_file(split_dict, out_dir):
    for k in ['train', 'val', 'test']:
        with open(os.path.join(out_dir, f"{k}.txt"), "w") as f:
            for item in split_dict[k]:
                f.write(f"{item['path']},{item['label']}\n")

def save_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

def main(root_dir, out_dir, weight_json=None, weight_dict=None, split_ratio=(0.8, 0.1, 0.1)):
    os.makedirs(out_dir, exist_ok=True)

    # 支持两种输入方式：json 文件路径 or 直接 dict
    if weight_dict:
        label_weights = weight_dict
    elif weight_json:
        with open(weight_json, 'r') as f:
            label_weights = json.load(f)
    else:
        raise ValueError("必须提供 weight_dict 或 weight_json")

    files_by_label, dur_by_label = collect_durations(root_dir)
    target_durations, limiting_label, max_total = compute_target_durations(dur_by_label, label_weights)

    print(f"限制标签：{limiting_label}，最大总时长为 {max_total:.2f} 秒")

    split_all = {"train": [], "val": [], "test": []}
    stat = {}

    for label in label_weights:
        selected, selected_dur = random_select(files_by_label[label], target_durations[label])
        train, val, test = split_train_val_test(selected, split_ratio)
        split_all["train"] += train
        split_all["val"] += val
        split_all["test"] += test
        stat[label] = {
            "target_sec": target_durations[label],
            "selected_sec": selected_dur,
            "train_sec": sum(f["duration"] for f in train),
            "val_sec": sum(f["duration"] for f in val),
            "test_sec": sum(f["duration"] for f in test),
            "train_count": len(train),
            "val_count": len(val),
            "test_count": len(test),
        }

    write_split_file(split_all, out_dir)
    save_summary(stat, os.path.join(out_dir, "split_summary.json"))
    print("✅ 划分完成！结果保存在", out_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--weight_json", required=True)
    args = parser.parse_args()
    main(args.root_dir, args.out_dir, args.weight_json)