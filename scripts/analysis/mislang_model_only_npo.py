import csv
import json
import os
import re
import urllib.request
from collections import Counter

import fasttext
import numpy as np


MODEL_FILE = "lid.176.ftz"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
NAME = "lacomsa"
RESPONSES_ROOT = f"responses/{NAME}"
OUTPUT_CSV = f"analysis/mislang_model_{NAME}_only_npo.csv"

FOLDER_PATTERN = re.compile(
    r"^(?P<lang_prefix>[a-z]+)-results-.*?_8b_(?P<model_type>.+?)_(?P<version>v\d+)$"
)


original_array = np.array


def patched_array(obj, **kwargs):
    if kwargs.get("copy") is False:
        kwargs.pop("copy")
    return original_array(obj, **kwargs)


np.array = patched_array


def ensure_model_exists():
    if not os.path.exists(MODEL_FILE):
        print(f"Downloading {MODEL_FILE}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Download complete.")


def parse_folder_metadata(folder_name):
    match = FOLDER_PATTERN.match(folder_name)
    if not match:
        return None
    return match.groupdict()


def should_keep_model(model_type):
    normalized = model_type.strip().lower()
    return normalized.startswith("npo")


def detect_lang(ft_model, text):
    labels, _ = ft_model.predict(text, k=1)
    return labels[0].replace("__label__", "")


def analyze_model_outputs_file(ft_model, model_outputs_path, expected_lang):
    with open(model_outputs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {"total": 0, "mismatch": 0}
    output_total_length = 0
    mismatch_detected_counter = Counter()

    for item in data:
        output_text = item.get("output")
        if not output_text:
            continue

        output_clean = str(output_text).replace("\n", " ")

        try:
            output_lang = detect_lang(ft_model, output_clean)
        except Exception:
            continue

        stats["total"] += 1
        output_total_length += len(output_clean)
        if output_lang != expected_lang:
            stats["mismatch"] += 1
            mismatch_detected_counter[output_lang] += 1

    return stats, output_total_length, mismatch_detected_counter


def build_row(folder_name, meta, stats, output_total_length, mismatch_detected_counter):
    total = stats["total"]
    mismatch = stats["mismatch"]
    mislang_pct = (mismatch / total * 100) if total else 0.0
    avg_length = (output_total_length / total) if total else 0.0
    top3 = mismatch_detected_counter.most_common(3)
    top3_total = sum(count for _, count in top3)
    others_count = max(0, mismatch - top3_total)
    others_pct = (others_count / mismatch * 100) if mismatch else 0.0

    row = {
        "folder": folder_name,
        "lang_prefix": meta["lang_prefix"],
        "model_type": meta["model_type"],
        "version": meta["version"],
        "total_samples": total,
        "mislang_count": mismatch,
        "mislang_pct": f"{mislang_pct:.2f}",
        "avg_length": f"{avg_length:.2f}",
        "top1_lang": "",
        "top1_count": "",
        "top1_pct": "",
        "top2_lang": "",
        "top2_count": "",
        "top2_pct": "",
        "top3_lang": "",
        "top3_count": "",
        "top3_pct": "",
        "others_count": others_count,
        "others_pct": f"{others_pct:.2f}",
    }

    for i, (lang, count) in enumerate(top3, start=1):
        pct = (count / mismatch * 100) if mismatch else 0.0
        row[f"top{i}_lang"] = lang
        row[f"top{i}_count"] = count
        row[f"top{i}_pct"] = f"{pct:.2f}"

    return row


def append_rows(rows):
    if not rows:
        print("No rows to append.")
        return

    fieldnames = list(rows[0].keys())
    csv_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0

    with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Appended {len(rows)} row(s) to: {OUTPUT_CSV}")


def main():
    ensure_model_exists()
    ft_model = fasttext.load_model(MODEL_FILE)

    rows = []
    for entry in sorted(os.scandir(RESPONSES_ROOT), key=lambda e: e.name):
        if not entry.is_dir():
            continue

        meta = parse_folder_metadata(entry.name)
        if not meta:
            continue
        if not should_keep_model(meta["model_type"]):
            continue

        model_outputs_path = os.path.join(entry.path, "model_outputs.json")
        if not os.path.exists(model_outputs_path):
            continue

        stats, output_total_length, mismatch_detected_counter = analyze_model_outputs_file(
            ft_model,
            model_outputs_path,
            meta["lang_prefix"],
        )
        row = build_row(
            entry.name,
            meta,
            stats,
            output_total_length,
            mismatch_detected_counter,
        )
        rows.append(row)
        print(
            f"Processed {entry.name}: mislang={row['mislang_pct']}%, "
            f"avg_length={row['avg_length']}"
        )

    append_rows(rows)


if __name__ == "__main__":
    main()
