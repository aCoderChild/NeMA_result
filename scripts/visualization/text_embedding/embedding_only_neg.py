import argparse
import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI


CACHE_SCHEMA = "icr_negative_rejected_v1"


def format_embedding_text(instruction: str, response: str) -> str:
    return f"### Instruction: {instruction} ### Response: {response}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Embed rejected responses from train_icr.jsonl (instruction + rejected) "
            "and write a cache file (no plot)."
        )
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="/home/gangstat/NeMA_result/_pairs/train_icr.jsonl",
        help="Path to train_icr.jsonl.",
    )
    parser.add_argument(
        "--start-line",
        type=int,
        default=5001,
        help="First line to read (1-based, inclusive).",
    )
    parser.add_argument(
        "--end-line",
        type=int,
        default=5030,
        help="Last line to read (1-based, inclusive).",
    )
    parser.add_argument(
        "--require-lang",
        type=str,
        default="ru",
        help="Only keep rows where lang matches (empty string = no filter).",
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
        default="/home/gangstat/NeMA_result/analysis/embeddings/icr_train_icr_ru_lines_5001_5030_neg_rejected_cache.json",
        help="Output cache JSON path.",
    )
    return parser.parse_args()


def text_key(embedding_text: str, embedding_model: str) -> str:
    return hashlib.sha256(f"{embedding_model}\n{embedding_text}".encode("utf-8")).hexdigest()


def load_cache(cache_path: Path):
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


def save_cache(cache_path: Path, cache):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache["cache_schema"] = CACHE_SCHEMA
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def upsert_cache_item(cache, item):
    for i, old in enumerate(cache["items"]):
        if old.get("cache_key") == item.get("cache_key"):
            cache["items"][i] = item
            return
    cache["items"].append(item)


def iter_jsonl_lines(path: Path, start_line: int, end_line: int):
    if start_line < 1 or end_line < start_line:
        raise ValueError(f"Invalid line range: {start_line}-{end_line}")
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no > end_line:
                break
            if line_no < start_line:
                continue
            yield line_no, json.loads(line)


def build_records(
    jsonl_path: Path,
    start_line: int,
    end_line: int,
    require_lang: str,
):
    records = []
    variant = f"train_icr_neg_rejected_{require_lang}_L{start_line}_L{end_line}"
    for line_no, row in iter_jsonl_lines(jsonl_path, start_line, end_line):
        if require_lang and row.get("lang") != require_lang:
            raise ValueError(
                f"Line {line_no}: expected lang={require_lang!r}, got {row.get('lang')!r}"
            )
        instruction = row.get("instruction") or ""
        rejected = row.get("rejected") or ""
        if not str(rejected).strip():
            raise ValueError(f"Line {line_no}: empty rejected field")
        embedding_text = format_embedding_text(instruction, rejected)
        records.append(
            {
                "group": "negative_rejected",
                "lang": row.get("lang", "unknown"),
                "variant": variant,
                "folder": jsonl_path.name,
                "output_field": "rejected",
                "jsonl_line": line_no,
                "pair_id": row.get("id"),
                "model_type": "train_icr_rejected",
                "instruction": instruction,
                "response": rejected,
                "embedding_text": embedding_text,
            }
        )
    expected = end_line - start_line + 1
    if len(records) != expected:
        raise ValueError(
            f"Expected {expected} lines in range, got {len(records)} "
            f"(file may be shorter than {end_line})"
        )
    return records


def fill_embeddings(records, embedding_model: str, cache_path: Path):
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

    cached_by_key = {it["cache_key"]: it for it in cache["items"] if "cache_key" in it}
    to_embed = []

    for rec in records:
        et = rec["embedding_text"]
        key = text_key(et, embedding_model)
        rec["cache_key"] = key
        hit = cached_by_key.get(key)
        if hit and "embedding" in hit:
            rec["embedding"] = hit["embedding"]
        else:
            to_embed.append(rec)

    if to_embed:
        client = OpenAI(api_key=api_key)
        inputs = [r["embedding_text"] for r in to_embed]
        resp = client.embeddings.create(model=embedding_model, input=inputs)
        for rec, row in zip(to_embed, resp.data):
            rec["embedding"] = row.embedding

    for rec in records:
        upsert_cache_item(
            cache,
            {
                "cache_key": rec["cache_key"],
                "group": rec["group"],
                "lang": rec["lang"],
                "variant": rec["variant"],
                "folder": rec["folder"],
                "output_field": rec["output_field"],
                "jsonl_line": rec["jsonl_line"],
                "pair_id": rec["pair_id"],
                "sample_id": None,
                "model_type": rec["model_type"],
                "embedding_model": embedding_model,
                "instruction": rec["instruction"],
                "response": rec["response"],
                "embedding_text": rec["embedding_text"],
                "text": rec["embedding_text"],
                "embedding": rec["embedding"],
            },
        )

    save_cache(cache_path, cache)


def main():
    args = parse_args()
    jsonl_path = Path(args.jsonl_path)
    cache_path = Path(args.cache_path)

    records = build_records(
        jsonl_path=jsonl_path,
        start_line=args.start_line,
        end_line=args.end_line,
        require_lang=args.require_lang.strip(),
    )
    fill_embeddings(records, args.embedding_model, cache_path)

    print(f"Saved cache: {cache_path}")
    print(f"schema: {CACHE_SCHEMA}")
    print(f"Lines: {args.start_line}-{args.end_line} lang={args.require_lang!r}")
    print(f"Rows: {len(records)}")


if __name__ == "__main__":
    main()
