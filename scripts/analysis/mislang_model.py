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
RESPONSES_ROOT = "responses/icr"
OUTPUT_CSV = "analysis/lang_correct_incorrect_distribution_model_icr.csv"

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
    if normalized.startswith("dpo") and normalized != "dpo_250426":
        return False
    return True


def detect_lang(ft_model, text):
    labels, _ = ft_model.predict(text, k=1)
    return labels[0].replace("__label__", "")


def analyze_model_outputs_file(ft_model, model_outputs_path, expected_lang):
    with open(model_outputs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Use total = number of entries in the file so missing/failed detections are
    # still counted and can be treated as incorrect.
    stats = {"total": 0, "correct": 0, "incorrect": 0, "no_responses": 0}
    output_total_length = 0
    mismatch_detected_counter = Counter()
    ground_truth_lang_counter = Counter()
    seen_records = set()

    for i, item in enumerate(data):
        record_key = (
            item.get("id"),
            item.get("instruction"),
            item.get("output"),
        )
        if record_key in seen_records:
            print(
                f"DEBUG: Duplicate record skipped at {model_outputs_path} index={i}, "
                f"id={item.get('id')}, expected_lang={expected_lang}"
            )
            continue
        seen_records.add(record_key)

        stats["total"] += 1
        output_text = item.get("output")

        if not output_text:
            # Missing output — count as no_response and log debug
            stats["no_responses"] += 1
            print(f"DEBUG: Missing output at {model_outputs_path} index={i}, expected_lang={expected_lang}")
            continue

        output_clean = str(output_text).replace("\n", " ")

        try:
            output_lang = detect_lang(ft_model, output_clean)
        except Exception as e:
            # Detection failed — count as no_response and log debug
            stats["no_responses"] += 1
            print(
                f"DEBUG: Language detection failed at {model_outputs_path} index={i}, "
                f"expected_lang={expected_lang}, error={e}"
            )
            continue

        output_total_length += len(output_clean)

        if output_lang == expected_lang:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1
            mismatch_detected_counter[output_lang] += 1
            ground_truth_lang_counter[expected_lang] += 1

    return stats, output_total_length, mismatch_detected_counter, ground_truth_lang_counter


def build_aggregated_rows(model_type_data):
    """Build rows aggregated by model_type and language with correct/incorrect counts."""
    rows = []
    
    for model_type in sorted(model_type_data.keys()):
        for lang in sorted(model_type_data[model_type].keys()):
            data = model_type_data[model_type][lang]
            total_incorrect = data["total_incorrect"]
            total_correct = data["total_correct"]
            total_no_responses = data.get("total_no_responses", 0)
            total_samples = total_correct + total_incorrect + total_no_responses

            row = {
                "model_type": model_type,
                "lang": lang,
                "total_samples": total_samples,
                "correct_count": total_correct,
                "incorrect_count": total_incorrect,
                "no_responses": total_no_responses,
            }
            
            rows.append(row)
    
    return rows


def write_aggregated_rows(rows, output_csv):
    """Write aggregated rows to CSV, replacing existing file."""
    if not rows:
        print("No rows to write.")
        return

    fieldnames = ["model_type", "lang", "total_samples", "correct_count", "incorrect_count", "no_responses"]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to: {output_csv}")


def main():
    ensure_model_exists()
    ft_model = fasttext.load_model(MODEL_FILE)

    # Aggregate data by model_type and language
    model_type_data = {}

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

        stats, output_total_length, mismatch_detected_counter, ground_truth_lang_counter = analyze_model_outputs_file(
            ft_model,
            model_outputs_path,
            meta["lang_prefix"],
        )
        
        # Initialize model_type and language entry if not present
        model_type = meta["model_type"]
        lang = meta["lang_prefix"]
        
        if model_type not in model_type_data:
            model_type_data[model_type] = {}
        
        if lang not in model_type_data[model_type]:
            model_type_data[model_type][lang] = {
                "total_correct": 0,
                "total_incorrect": 0,
                "total_no_responses": 0,
                "ground_truth_lang_counter": Counter(),
            }
        
        # Aggregate data for this model_type and language
        model_type_data[model_type][lang]["total_correct"] += stats["correct"]
        model_type_data[model_type][lang]["total_incorrect"] += stats["incorrect"]
        model_type_data[model_type][lang]["total_no_responses"] += stats.get("no_responses", 0)
        model_type_data[model_type][lang]["ground_truth_lang_counter"].update(ground_truth_lang_counter)
        
        print(
            f"Processed {entry.name}: correct={stats['correct']}, incorrect={stats['incorrect']}"
        )

    rows = build_aggregated_rows(model_type_data)
    write_aggregated_rows(rows, OUTPUT_CSV)


if __name__ == "__main__":
    main()
