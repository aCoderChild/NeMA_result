import csv
import json
import os
import sys
import urllib.request
from collections import Counter, defaultdict

import fasttext
import numpy as np


MODEL_FILE = "lid.176.ftz"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
DATA_DIR = "/home/gangstat/NeMA_result/_pairs"
OUTPUT_DIR = "/home/gangstat/NeMA_result/analysis"
TARGET_EXPECTED_LANGS = ["es", "ru"]
TOP_DETECTED_LANGS = 8


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


def build_confusion_all_samples(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None

    ensure_model_exists()
    ft_model = fasttext.load_model(MODEL_FILE)

    # expected_lang -> Counter(detected_lang -> sample_count)
    matrix_counts = defaultdict(Counter)
    total_by_expected = Counter()

    print(f"Scanning file: {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                expected_lang = data.get("lang")
                instruction_text = data.get("instruction")
                chosen_text = data.get("chosen")
                if not chosen_text:
                    continue

                # Fallback for datasets where `lang` is missing/unknown (e.g. "unk"):
                # infer expected language from the instruction.
                if not expected_lang or expected_lang == "unk":
                    if not instruction_text:
                        continue
                    instruction_clean = str(instruction_text).replace("\n", " ")
                    instruction_labels, _ = ft_model.predict(instruction_clean, k=1)
                    expected_lang = instruction_labels[0].replace("__label__", "")

                if expected_lang not in TARGET_EXPECTED_LANGS:
                    continue

                text_clean = str(chosen_text).replace("\n", " ")
                labels, _ = ft_model.predict(text_clean, k=1)
                detected_lang = labels[0].replace("__label__", "")

                matrix_counts[expected_lang][detected_lang] += 1
                total_by_expected[expected_lang] += 1
            except Exception:
                continue

    if not total_by_expected:
        print("No valid samples found for target expected languages.")
        return [], []

    return matrix_counts, total_by_expected


def export_confusion_csv(matrix_counts, total_by_expected, output_csv_path):
    if matrix_counts is None or total_by_expected is None:
        return

    detected_totals = Counter()
    for expected_lang in TARGET_EXPECTED_LANGS:
        detected_totals.update(matrix_counts.get(expected_lang, Counter()))

    base_detected_langs = ["es", "ru"]
    extra_slots = max(0, TOP_DETECTED_LANGS - len(base_detected_langs))
    top_other_detected_langs = [
        lang
        for lang, _ in detected_totals.most_common()
        if lang not in base_detected_langs
    ][:extra_slots]
    selected_detected_langs = base_detected_langs + top_other_detected_langs

    rows = []
    for expected_lang in TARGET_EXPECTED_LANGS:
        row = {"expected_lang": expected_lang}
        total_count = total_by_expected.get(expected_lang, 0)

        others_count = 0
        for detected_lang, count in matrix_counts.get(expected_lang, Counter()).items():
            if detected_lang not in selected_detected_langs:
                others_count += count

        for detected_lang in selected_detected_langs:
            count = matrix_counts.get(expected_lang, Counter()).get(detected_lang, 0)
            pct = (count / total_count * 100) if total_count else 0.0
            row[detected_lang] = f"{pct:.2f}" if pct > 0 else ""

        others_pct = (others_count / total_count * 100) if total_count else 0.0
        row["others"] = f"{others_pct:.2f}" if others_pct > 0 else ""
        rows.append(row)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    fieldnames = ["expected_lang"] + selected_detected_langs + ["others"]

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Confusion matrix exported to: {output_csv_path}")


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "train_mapo.jsonl"
    input_path = os.path.join(DATA_DIR, filename)

    output_name = f"confusion_{os.path.splitext(filename)[0]}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    matrix_counts, total_by_expected = build_confusion_all_samples(input_path)
    export_confusion_csv(matrix_counts, total_by_expected, output_path)
