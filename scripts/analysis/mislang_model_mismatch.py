import argparse
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
RESPONSES_ROOTS = ["responses/icr", "responses/mapo", "responses/lacomsa"]
OUTPUT_CSV = "analysis/mislang_multilangs_model_mapo.csv"
LANGUAGES = ["de", "fr", "en", "es", "ru"]
DEFAULT_METHOD = "mapo"

FILE_PATTERN = re.compile(
    r"^(?P<lang_prefix>[a-z]+)-results-(?P<method>[a-z]+)_enesru_llama3_8b_(?P<model_type>.+?)_(?P<version>v\d+)$"
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


def parse_file_metadata(file_name):
    match = FILE_PATTERN.match(file_name)
    if not match:
        return None
    return match.groupdict()


def detect_lang(ft_model, text):
    labels, _ = ft_model.predict(text, k=1)
    return labels[0].replace("__label__", "")


def analyze_baseline_jsonl_file(ft_model, json_path, expected_lang):
    stats = {"predicted": 0}
    predicted_lang_counter = Counter()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    for item in data:
        output_field = item.get("output")
        output_text = None
        if isinstance(output_field, list) and output_field:
            output_text = output_field[0]
        elif isinstance(output_field, str):
            output_text = output_field

        if not output_text:
            print(f"DEBUG: Missing output in {json_path} for expected_lang={expected_lang}")
            continue

        output_clean = str(output_text).replace("\n", " ")

        try:
            output_lang = detect_lang(ft_model, output_clean)
        except Exception:
            print(f"DEBUG: Language detection failed in {json_path} for expected_lang={expected_lang}")
            continue

        stats["predicted"] += 1
        predicted_lang_counter[output_lang] += 1

    return stats, predicted_lang_counter


def build_row(meta, stats, predicted_lang_counter):
    predicted_total = stats["predicted"]
    others = max(0, predicted_total - sum(predicted_lang_counter.get(lang, 0) for lang in LANGUAGES))

    row = {
        "method": meta["method"],
        "model_type": meta["model_type"],
        "expected_lang": meta["lang_prefix"],
    }

    for lang in LANGUAGES:
        row[lang] = predicted_lang_counter.get(lang, 0)

    row["others"] = others

    return row


def write_rows(rows):
    if not rows:
        print("No rows to append.")
        return

    fieldnames = ["method", "model_type", "expected_lang", "de", "fr", "en", "es", "ru", "others"]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to: {OUTPUT_CSV}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build mismatch counts for a single method."
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=["icr", "mapo", "lacomsa"],
        help="Method to include in the CSV output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_model_exists()
    ft_model = fasttext.load_model(MODEL_FILE)

    rows = []
    for responses_root in RESPONSES_ROOTS:
        if os.path.basename(responses_root) != args.method:
            continue

        for entry in sorted(os.scandir(responses_root), key=lambda e: e.name):
            if not entry.is_dir():
                continue

            meta = parse_file_metadata(entry.name)
            if not meta:
                continue

            model_outputs_path = os.path.join(entry.path, "model_outputs.json")
            if not os.path.exists(model_outputs_path):
                continue

            stats, predicted_lang_counter = analyze_baseline_jsonl_file(
                ft_model,
                model_outputs_path,
                meta["lang_prefix"],
            )
            row = build_row(
                meta,
                stats,
                predicted_lang_counter,
            )
            rows.append(row)
            print(
                f"Processed {responses_root}/{entry.name}: method={row['method']}, expected_lang={row['expected_lang']}"
            )

    write_rows(rows)


if __name__ == "__main__":
    main()
