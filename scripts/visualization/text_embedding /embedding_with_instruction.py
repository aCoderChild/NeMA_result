import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from openai import OpenAI


EXPECTED_LANGS = ["de", "en", "es", "fr", "ru"]
EXPECTED_NPO_VARIANTS = [
    "npo_150426",
    "npo_checkpoint-1",
    "npo_checkpoint-2",
    "npo_checkpoint-3",
    "npo_checkpoint-4",
    "npo_checkpoint-5",
    "npo_checkpoint-10",
    "npo_checkpoint-20",
    "npo_checkpoint-30",
]

CACHE_SCHEMA = "instruction_plus_response_v1"


def format_embedding_text(instruction: str, response: str) -> str:
    return f"### Instruction: {instruction} ### Response: {response}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch OpenAI embeddings for ICR annotations using "
            "'### Instruction: ... ### Response: ...' text. Writes cache only (no plot)."
        )
    )
    parser.add_argument(
        "--icr-root",
        type=str,
        default="/home/gangstat/NeMA_result/responses/icr",
        help="Root directory containing icr result folders.",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=1,
        help="Sample id in each annotations.json.",
    )
    parser.add_argument(
        "--base-from-variant",
        type=str,
        default="npo_150426",
        help="NPO variant folder used to read base output_1 per language.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/embeddings/icr_id1_embeddings_with_instruction_cache.json",
        help="Path to save/reuse cache.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="/home/gangstat/NeMA_result/responses/baseline",
        help="Directory with per-lang baseline JSONL files.",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help=(
            "Rebuild ICR (GPT-4 base + all NPO) and baseline embeddings. "
            "By default, if cache already exists and has items, only baseline rows are embedded/upserted."
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not embed baseline (ICR only). Useful with --force-full if baseline added later.",
    )
    return parser.parse_args()


def extract_lang(folder_name):
    m = re.match(r"^([a-z]{2})-results-", folder_name)
    return m.group(1) if m else None


def extract_npo_variant(folder_name):
    if "npo_150426" in folder_name:
        return "npo_150426"
    m = re.search(r"npo_checkpoint-(\d+)", folder_name)
    if m:
        return f"npo_checkpoint-{m.group(1)}"
    return None


def load_annotations_sample(annotations_path, sample_id):
    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        row_id = row.get("id")
        if row_id == sample_id or str(row_id) == str(sample_id):
            return row
    raise ValueError(f"id={sample_id} not found in {annotations_path}")


def baseline_file_for_lang(baseline_dir, lang):
    matches = sorted(
        baseline_dir.glob(
            f"{lang}.json.prediction.with_Llama-3-Base-8B-SFT-DPO.to_{lang}.jsonl"
        )
    )
    if not matches:
        raise FileNotFoundError(
            f"Baseline JSONL not found for lang={lang} in {baseline_dir}"
        )
    return matches[0]


def read_jsonl_record_by_id(path, sample_id):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row_id = row.get("id")
            if row_id == sample_id or str(row_id) == str(sample_id):
                return row
    raise ValueError(f"id={sample_id} not found in {path}")


def build_baseline_records(baseline_dir, sample_id):
    records = []
    baseline_dir = Path(baseline_dir)
    for lang in EXPECTED_LANGS:
        fpath = baseline_file_for_lang(baseline_dir, lang)
        row = read_jsonl_record_by_id(fpath, sample_id)
        instruction = row.get("instruction") or ""
        responses = row.get("response", [])
        if not responses:
            raise ValueError(f"Empty response list in {fpath} for id={sample_id}")
        response = responses[0] if isinstance(responses[0], str) else str(responses[0])
        embedding_text = format_embedding_text(instruction, response)
        records.append(
            {
                "group": "baseline_output",
                "lang": lang,
                "variant": "baseline_sft_dpo",
                "folder": fpath.name,
                "output_field": "response[0]",
                "sample_id": sample_id,
                "model_type": "Llama-3-Base-8B-SFT-DPO",
                "instruction": instruction,
                "response": response,
                "embedding_text": embedding_text,
            }
        )
    for r in records:
        if not str(r["response"]).strip():
            raise ValueError(
                f"Empty baseline response (lang={r['lang']}, id={sample_id})"
            )
    return records


def find_npo_folders(icr_root):
    variant_lang_folder = {}
    for p in icr_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        lang = extract_lang(name)
        if not lang:
            continue
        variant = extract_npo_variant(name)
        if not variant:
            continue
        if not (p / "annotations.json").exists():
            continue
        variant_lang_folder.setdefault(variant, {})[lang] = p
    return variant_lang_folder


def text_key(embedding_text, embedding_model):
    return hashlib.sha256(f"{embedding_model}\n{embedding_text}".encode("utf-8")).hexdigest()


def load_cache(cache_path):
    if not cache_path.exists():
        return {
            "embedding_model": None,
            "cache_schema": CACHE_SCHEMA,
            "items": [],
        }
    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("cache_schema", CACHE_SCHEMA)
    data.setdefault("items", [])
    return data


def save_cache(cache_path, cache):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache["cache_schema"] = CACHE_SCHEMA
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def upsert_cache_item(cache, item):
    items = cache["items"]
    for i, old in enumerate(items):
        if old["cache_key"] == item["cache_key"]:
            items[i] = item
            return
    items.append(item)


def build_records(icr_root, sample_id, base_from_variant):
    records = []
    npo_map = find_npo_folders(icr_root)

    if base_from_variant not in npo_map:
        raise ValueError(
            f"Base variant '{base_from_variant}' not found. "
            f"Available variants: {sorted(npo_map.keys())}"
        )

    for lang in EXPECTED_LANGS:
        folder = npo_map[base_from_variant].get(lang)
        if folder is None:
            raise ValueError(f"Missing lang={lang} in base variant={base_from_variant}")
        sample = load_annotations_sample(folder / "annotations.json", sample_id)
        instruction = sample.get("instruction") or ""
        response = sample.get("output_1") or ""
        embedding_text = format_embedding_text(instruction, response)
        records.append(
            {
                "group": "base_output_1",
                "lang": lang,
                "variant": "base_output_1",
                "folder": folder.name,
                "output_field": "output_1",
                "sample_id": sample_id,
                "model_type": sample.get("generator_1", "unknown"),
                "instruction": instruction,
                "response": response,
                "embedding_text": embedding_text,
            }
        )

    missing_variants = [v for v in EXPECTED_NPO_VARIANTS if v not in npo_map]
    if missing_variants:
        raise ValueError(f"Missing NPO variants: {missing_variants}")

    for variant in EXPECTED_NPO_VARIANTS:
        for lang in EXPECTED_LANGS:
            folder = npo_map[variant].get(lang)
            if folder is None:
                raise ValueError(f"Missing lang={lang} for NPO variant={variant}")
            sample = load_annotations_sample(folder / "annotations.json", sample_id)
            instruction = sample.get("instruction") or ""
            response = sample.get("output_2") or ""
            embedding_text = format_embedding_text(instruction, response)
            records.append(
                {
                    "group": "npo_output_2",
                    "lang": lang,
                    "variant": variant,
                    "folder": folder.name,
                    "output_field": "output_2",
                    "sample_id": sample_id,
                    "model_type": sample.get("generator_2", "unknown"),
                    "instruction": instruction,
                    "response": response,
                    "embedding_text": embedding_text,
                }
            )

    for r in records:
        if not r["response"].strip():
            raise ValueError(
                f"Empty response ({r['folder']}, {r['output_field']}, id={sample_id})"
            )
    return records


def fill_embeddings(records, embedding_model, cache_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    cache = load_cache(cache_path)
    if cache.get("embedding_model") not in (None, embedding_model):
        raise ValueError(
            f"Cache has embedding_model={cache.get('embedding_model')!r}, "
            f"requested {embedding_model!r}. Use a new --cache-path or matching model."
        )
    cache["embedding_model"] = embedding_model

    cached_by_key = {item["cache_key"]: item for item in cache["items"]}
    to_embed = []

    for rec in records:
        et = rec["embedding_text"]
        key = text_key(et, embedding_model)
        rec["cache_key"] = key
        cached = cached_by_key.get(key)
        if cached and "embedding" in cached:
            rec["embedding"] = cached["embedding"]
        else:
            to_embed.append(rec)

    if to_embed:
        client = OpenAI(api_key=api_key)
        inputs = [r["embedding_text"] for r in to_embed]
        response = client.embeddings.create(model=embedding_model, input=inputs)
        for rec, row in zip(to_embed, response.data):
            rec["embedding"] = row.embedding

    for rec in records:
        cache_item = {
            "cache_key": rec["cache_key"],
            "group": rec["group"],
            "lang": rec["lang"],
            "variant": rec["variant"],
            "folder": rec["folder"],
            "output_field": rec["output_field"],
            "sample_id": rec["sample_id"],
            "model_type": rec["model_type"],
            "embedding_model": embedding_model,
            "instruction": rec["instruction"],
            "response": rec["response"],
            "embedding_text": rec["embedding_text"],
            "text": rec["embedding_text"],
            "embedding": rec["embedding"],
        }
        upsert_cache_item(cache, cache_item)

    save_cache(cache_path, cache)
    return records


def main():
    args = parse_args()
    icr_root = Path(args.icr_root)
    cache_path = Path(args.cache_path)
    baseline_dir = Path(args.baseline_dir)

    cache_pre = load_cache(cache_path) if cache_path.exists() else None
    has_nonempty_cache = (
        cache_pre is not None and len(cache_pre.get("items", [])) > 0
    )
    baseline_only = has_nonempty_cache and not args.force_full

    if baseline_only:
        print(
            "Cache exists with items → incremental mode: embedding baseline only "
            "(use --force-full to rebuild ICR + baseline)."
        )
        records_bl = build_baseline_records(baseline_dir, args.sample_id)
        fill_embeddings(
            records=records_bl,
            embedding_model=args.embedding_model,
            cache_path=cache_path,
        )
        print(f"Saved cache: {cache_path}")
        print(f"schema: {CACHE_SCHEMA}")
        print(f"sample_id: {args.sample_id}")
        print(f"Mode: baseline_only, rows updated: {len(records_bl)}")
        return

    records_icr = build_records(
        icr_root=icr_root,
        sample_id=args.sample_id,
        base_from_variant=args.base_from_variant,
    )
    fill_embeddings(
        records=records_icr,
        embedding_model=args.embedding_model,
        cache_path=cache_path,
    )
    total = len(records_icr)

    if not args.skip_baseline:
        records_bl = build_baseline_records(baseline_dir, args.sample_id)
        fill_embeddings(
            records=records_bl,
            embedding_model=args.embedding_model,
            cache_path=cache_path,
        )
        total += len(records_bl)
    else:
        print("Skipped baseline (--skip-baseline).")

    print(f"Saved cache: {cache_path}")
    print(f"schema: {CACHE_SCHEMA}")
    print(f"sample_id: {args.sample_id}")
    print(f"Mode: full (ICR + baseline unless --skip-baseline), rows processed: {total}")


if __name__ == "__main__":
    main()
