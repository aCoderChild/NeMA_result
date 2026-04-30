import csv
import json
import os
import sys
import urllib.request
from collections import Counter

import fasttext
import numpy as np


MODEL_FILE = "lid.176.ftz"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
DATA_DIR = "/home/gangstat/NeMA_result/_pairs"
OUTPUT_CSV = "/home/gangstat/NeMA_result/analysis/mislang_data.csv"


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


def analyze_mislang(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    ensure_model_exists()
    ft_model = fasttext.load_model(MODEL_FILE)

    stats = {"total": 0, "match": 0, "mismatch": 0, "errors": 0}
    details = []
    mismatch_detected_lang_counter = Counter()
    total_response_length = 0

    print(f"Scanning file: {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                target_lang = data.get("lang")
                rejected_text = data.get("rejected")

                if target_lang and rejected_text:
                    text_clean = str(rejected_text).replace("\n", " ")
                    total_response_length += len(text_clean)
                    labels, _ = ft_model.predict(text_clean, k=1)
                    pred_lang = labels[0].replace("__label__", "")

                    stats["total"] += 1
                    if pred_lang == target_lang:
                        stats["match"] += 1
                    else:
                        stats["mismatch"] += 1
                        mismatch_detected_lang_counter[pred_lang] += 1
                        if len(details) < 5:
                            details.append(
                                {
                                    "line": line_idx + 1,
                                    "expected": target_lang,
                                    "detected": pred_lang,
                                    "text": text_clean[:100],
                                }
                            )
                else:
                    stats["errors"] += 1
            except Exception:
                stats["errors"] += 1

    print("\n" + "=" * 50)
    print("FINAL ANALYSIS RESULTS")
    print("=" * 50)
    if stats["total"] > 0:
        print(
            f"Language match:    {stats['match']} "
            f"({(stats['match'] / stats['total']) * 100:.2f}%)"
        )
        print(
            f"Language mismatch: {stats['mismatch']} "
            f"({(stats['mismatch'] / stats['total']) * 100:.2f}%)"
        )
        print(f"Invalid/missing:   {stats['errors']}")
        print(f"Total samples:     {stats['total']}")

        if details:
            print("\nMismatch examples:")
            for d in details:
                print(
                    f"  - Line {d['line']}: expected {d['expected']} "
                    f"but detected {d['detected']}"
                )
                print(f"    Text: {d['text']}...")
    else:
        print("No valid records processed. Check the 'lang' and 'rejected' keys.")

    append_result_to_csv(
        file_path,
        stats,
        mismatch_detected_lang_counter,
        total_response_length,
    )


def append_result_to_csv(
    file_path,
    stats,
    mismatch_detected_lang_counter,
    total_response_length,
):
    mismatch_pct = (stats["mismatch"] / stats["total"] * 100) if stats["total"] else 0.0
    avg_response_length = (
        total_response_length / stats["total"] if stats["total"] else 0.0
    )
    top3 = mismatch_detected_lang_counter.most_common(3)

    row = {
        "filename": os.path.basename(file_path),
        "total_samples": stats["total"],
        "mismatch_count": stats["mismatch"],
        "mislang_pct": f"{mismatch_pct:.2f}",
        "avg_response_length": f"{avg_response_length:.2f}",
        "top1_lang": "",
        "top1_count": "",
        "top1_pct": "",
        "top2_lang": "",
        "top2_count": "",
        "top2_pct": "",
        "top3_lang": "",
        "top3_count": "",
        "top3_pct": "",
    }

    for i, (lang, count) in enumerate(top3, start=1):
        pct = (count / stats["mismatch"] * 100) if stats["mismatch"] else 0.0
        row[f"top{i}_lang"] = lang
        row[f"top{i}_count"] = count
        row[f"top{i}_pct"] = f"{pct:.2f}"

    fieldnames = list(row.keys())
    csv_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0

    with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended result to CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "train_mapo.jsonl"
    file_path = os.path.join(DATA_DIR, filename)
    analyze_mislang(file_path)
