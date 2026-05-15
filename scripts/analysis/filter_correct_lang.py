"""
Filter output_gen jsonl files to keep only responses in the correct language.

Reads responses/output_gen/{lang}.jsonl, detects each response's language via
fasttext, and writes only matching entries to responses/output_gen_filtered/.
Also writes a CSV counting kept samples per instruction_id per language.
"""

import csv
import json
import os
import urllib.request
from collections import defaultdict

import fasttext
import numpy as np


MODEL_FILE = "lid.176.ftz"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
INPUT_DIR = "responses/output_gen"
OUTPUT_DIR = "responses/output_gen_filtered"

LANGUAGES = ["de", "en", "es", "fr", "ru"]


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


def detect_lang(ft_model, text):
    labels, _ = ft_model.predict(text, k=1)
    return labels[0].replace("__label__", "")


def filter_file(ft_model, lang, input_path, output_path):
    kept = 0
    skipped = 0
    counts = defaultdict(int)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            response = entry.get("response", "")
            if not response:
                skipped += 1
                continue

            response_clean = str(response).replace("\n", " ")
            try:
                detected = detect_lang(ft_model, response_clean)
            except Exception:
                skipped += 1
                continue

            if detected == lang:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                counts[entry.get("instruction_id", "")] += 1
                kept += 1
            else:
                skipped += 1

    return kept, skipped, counts


def write_counts_csv(all_counts, output_path):
    instruction_ids = sorted({iid for lang_counts in all_counts.values() for iid in lang_counts})

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction_id"] + LANGUAGES)
        writer.writeheader()
        for iid in instruction_ids:
            row = {"instruction_id": iid}
            for lang in LANGUAGES:
                row[lang] = all_counts[lang].get(iid, 0)
            writer.writerow(row)

    print(f"Counts CSV written to: {output_path}")


def main():
    ensure_model_exists()
    ft_model = fasttext.load_model(MODEL_FILE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_counts = {}

    for lang in LANGUAGES:
        input_path = os.path.join(INPUT_DIR, f"{lang}.jsonl")
        output_path = os.path.join(OUTPUT_DIR, f"{lang}.jsonl")

        if not os.path.exists(input_path):
            print(f"[{lang}] Input file not found: {input_path}")
            all_counts[lang] = {}
            continue

        print(f"[{lang}] Filtering {input_path} ...")
        kept, skipped, counts = filter_file(ft_model, lang, input_path, output_path)
        all_counts[lang] = counts
        total = kept + skipped
        print(f"[{lang}] Kept {kept}/{total} ({kept/total*100:.1f}%), "
              f"removed {skipped} mislang responses -> {output_path}")

    write_counts_csv(all_counts, os.path.join(OUTPUT_DIR, "instruction_id_counts.csv"))


if __name__ == "__main__":
    main()
