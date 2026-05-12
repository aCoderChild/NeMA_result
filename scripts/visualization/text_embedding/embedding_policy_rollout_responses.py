"""
Embed policy rollout JSONL rows (instruction + first response string) into a JSON cache.

Same embedding text format as embedding_only_neg.py (### Instruction / ### Response).
Scans fixed response directories for *.jsonl; language is taken from the filename stem.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from openai import OpenAI


CACHE_SCHEMA = "icr_policy_rollout_response_v1"

DEFAULT_RESPONSE_DIRS = (
    "/home/gangstat/NeMA_result/responses/icr_ppo",
    "/home/gangstat/NeMA_result/responses/icr-w-reinforce-0.1-checkpoint10",
    "/home/gangstat/NeMA_result/responses/icr-w-reinforce-0.1-checkpoint36",
)


def format_embedding_text(instruction: str, response: str) -> str:
    return f"### Instruction: {instruction} ### Response: {response}"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Embed rows from policy rollout JSONL files (instruction + first response) "
            "and write one JSON cache (same layout as embedding_only_neg.py)."
        )
    )
    p.add_argument(
        "--response-dirs",
        type=str,
        default=",".join(DEFAULT_RESPONSE_DIRS),
        help="Comma-separated directories containing <lang>.jsonl rollout files.",
    )
    p.add_argument(
        "--langs",
        type=str,
        default="de,ru",
        help="Only include JSONL stems in this comma-separated set (e.g. de,ru). Empty = all *.jsonl.",
    )
    p.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model.",
    )
    p.add_argument(
        "--cache-path",
        type=str,
        default="/home/gangstat/NeMA_result/analysis/embeddings/"
        "icr_policy_rollouts_ppo_reinforce_cache.json",
        help="Output cache JSON path.",
    )
    return p.parse_args()


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


def _first_response_text(row: dict, jsonl_path: Path, line_no: int) -> str:
    r = row.get("response")
    if isinstance(r, list):
        if not r:
            raise ValueError(f"{jsonl_path}:{line_no}: empty response list")
        return str(r[0])
    if isinstance(r, str) and r.strip():
        return r
    raise ValueError(f"{jsonl_path}:{line_no}: missing or invalid response field")


def collect_records(response_dirs: list[Path], lang_filter: set[str]) -> list[dict]:
    records = []
    for d in response_dirs:
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")
        policy_name = d.name
        for jsonl_path in sorted(d.glob("*.jsonl")):
            lang = jsonl_path.stem.lower()
            if lang_filter and lang not in lang_filter:
                continue
            with jsonl_path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    instruction = row.get("instruction") or ""
                    resp = _first_response_text(row, jsonl_path, line_no)
                    if not str(instruction).strip():
                        raise ValueError(f"{jsonl_path}:{line_no}: empty instruction")
                    embedding_text = format_embedding_text(instruction, resp)
                    variant = f"{policy_name}|{jsonl_path.name}|L{line_no}"
                    records.append(
                        {
                            "group": "policy_rollout",
                            "lang": lang,
                            "variant": variant,
                            "folder": jsonl_path.name,
                            "output_field": "response",
                            "jsonl_line": line_no,
                            "pair_id": row.get("instruction_id") or row.get("id"),
                            "model_type": policy_name,
                            "instruction": instruction,
                            "response": resp,
                            "embedding_text": embedding_text,
                            "jsonl_path": str(jsonl_path.resolve()),
                        }
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
                "jsonl_path": rec["jsonl_path"],
            },
        )

    save_cache(cache_path, cache)


def main():
    args = parse_args()
    dirs = [Path(p.strip()) for p in args.response_dirs.split(",") if p.strip()]
    raw_langs = (args.langs or "").strip()
    lang_filter = {x.strip().lower() for x in raw_langs.split(",") if x.strip()} if raw_langs else set()

    records = collect_records(dirs, lang_filter)
    if not records:
        raise RuntimeError("No JSONL rows collected; check --response-dirs and --langs.")

    cache_path = Path(args.cache_path)
    fill_embeddings(records, args.embedding_model, cache_path)

    print(f"Saved cache: {cache_path}")
    print(f"schema: {CACHE_SCHEMA}")
    print(f"Rows embedded: {len(records)}")
    for name, n in sorted(Counter(r["model_type"] for r in records).items()):
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
