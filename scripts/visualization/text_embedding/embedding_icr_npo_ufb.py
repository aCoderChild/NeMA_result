"""
Embed first JSONL row per lang (default es, ru) for each ICR NPO ultrafeedback checkpoint.

Layout (per checkpoint):
  <samples_root>/checkpoint-<N>/multilingual_generate/ultrafeedback_binarized/subset/random_100/
    es*.jsonl  ru*.jsonl  ...
First line of each lang file: instruction + response[0] (or response string) → OpenAI embedding.
Writes one merged JSON cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from openai import OpenAI


CACHE_SCHEMA = "icr_npo_ultrafeedback_first_row_es_ru_v1"


def format_embedding_text(instruction: str, response: str) -> str:
    return f"### Instruction: {instruction} ### Response: {response}"


def text_key(embedding_text: str, embedding_model: str) -> str:
    return hashlib.sha256(f"{embedding_model}\n{embedding_text}".encode("utf-8")).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--samples-root",
        default="/home/gangstat/NeMA_result/responses/icr_npo_ultrafeeback_samples",
        help="Parent folder containing checkpoint-* dirs.",
    )
    p.add_argument(
        "--subset-rel-path",
        default="multilingual_generate/ultrafeedback_binarized/subset/random_100",
        help="Path under each checkpoint-* to the JSONL files.",
    )
    p.add_argument(
        "--langs",
        default="es,ru",
        help="For each checkpoint, embed first row of <lang>*.jsonl in this subset folder.",
    )
    p.add_argument(
        "--skip-checkpoints",
        default="20",
        help=(
            "Comma-separated checkpoint indices to skip (no embedding), e.g. 20 when data is missing. "
            "Use empty string to skip none."
        ),
    )
    p.add_argument(
        "--response-index",
        type=int,
        default=0,
        help="If 'response' is a list (n-sample completions), take this index.",
    )
    p.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
    )
    p.add_argument(
        "--cache-path",
        default="/home/gangstat/NeMA_result/analysis/embeddings/icr_npo_ultrafeedback_first_es_ru_cache.json",
    )
    p.add_argument(
        "--cache-only",
        action="store_true",
        help="Do not call OpenAI; all rows must already be in cache.",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-embed every row even if cache_key matches.",
    )
    return p.parse_args()


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {"cache_schema": CACHE_SCHEMA, "embedding_model": None, "items": []}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("cache_schema", CACHE_SCHEMA)
    data.setdefault("items", [])
    return data


def save_cache(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    data["cache_schema"] = CACHE_SCHEMA
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def upsert(cache_items: list, item: dict):
    ck = item.get("cache_key")
    if not ck:
        cache_items.append(item)
        return
    for i, old in enumerate(cache_items):
        if old.get("cache_key") == ck:
            cache_items[i] = item
            return
    cache_items.append(item)


def checkpoint_sort_key(name: str) -> tuple[int, str]:
    m = re.match(r"^checkpoint-(\d+)$", name)
    if m:
        return (0, f"{int(m.group(1)):08d}")
    return (1, name)


def list_checkpoint_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    dirs.sort(key=lambda p: checkpoint_sort_key(p.name))
    if not dirs:
        raise RuntimeError(f"No checkpoint-* folders under {root}")
    return dirs


def first_jsonl_row(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"Empty JSONL: {path}")


def pick_lang_jsonl(subset_dir: Path, lang: str) -> Path:
    matches = sorted(subset_dir.glob(f"{lang}*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No {lang}*.jsonl under {subset_dir}")
    return matches[0]


def response_text_from_row(row: dict, index: int) -> str:
    r = row.get("response")
    if isinstance(r, list):
        if not r or index < 0 or index >= len(r):
            raise ValueError(f"response list missing or index {index} out of range")
        return str(r[index]).strip()
    if isinstance(r, str):
        return r.strip()
    raise ValueError("Row has no string/list 'response'")


def parse_skip_checkpoint_nums(s: str) -> set[int]:
    s = (s or "").strip()
    if not s:
        return set()
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def collect_rows(
    samples_root: Path,
    subset_rel: Path,
    langs: set[str],
    response_index: int,
    skip_checkpoint_nums: set[int],
) -> list[dict]:
    rows_out = []
    for ckpt_dir in list_checkpoint_dirs(samples_root):
        m_ck = re.match(r"^checkpoint-(\d+)$", ckpt_dir.name)
        if m_ck and int(m_ck.group(1)) in skip_checkpoint_nums:
            print(f"[skip] {ckpt_dir.name} (--skip-checkpoints)")
            continue
        subset = ckpt_dir / subset_rel
        if not subset.is_dir():
            raise FileNotFoundError(f"Missing subset dir: {subset}")
        for lang in sorted(langs):
            jf = pick_lang_jsonl(subset, lang)
            row = first_jsonl_row(jf)
            instruction = (row.get("instruction") or "").strip()
            if not instruction:
                raise ValueError(f"{jf}: empty instruction")
            resp = response_text_from_row(row, response_index)
            et = format_embedding_text(instruction, resp)
            rows_out.append(
                {
                    "checkpoint_dir": ckpt_dir.name,
                    "lang": lang,
                    "jsonl_path": str(jf),
                    "jsonl_basename": jf.name,
                    "instruction": row.get("instruction") or "",
                    "instruction_id": row.get("instruction_id") or row.get("id"),
                    "response_used": resp,
                    "response_index": response_index,
                    "embedding_text": et,
                }
            )
    return rows_out


def embed_openai(texts: list[str], model: str) -> list[list[float]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    out = client.embeddings.create(model=model, input=texts)
    return [x.embedding for x in out.data]


def main():
    args = parse_args()
    langs = {x.strip() for x in args.langs.split(",") if x.strip()}
    if not langs:
        raise ValueError("--langs empty")

    samples_root = Path(args.samples_root)
    subset_rel = Path(args.subset_rel_path)
    cache_path = Path(args.cache_path)
    skip_nums = parse_skip_checkpoint_nums(args.skip_checkpoints)

    points = collect_rows(
        samples_root, subset_rel, langs, args.response_index, skip_nums
    )
    cache = load_cache(cache_path)
    if cache.get("embedding_model") not in (None, args.embedding_model) and not args.force_refresh:
        raise ValueError(
            f"Cache embedding_model {cache.get('embedding_model')!r} != {args.embedding_model!r}"
        )
    cache["embedding_model"] = args.embedding_model

    by_key = {it["cache_key"]: it for it in cache["items"] if it.get("cache_key")}
    to_embed: list[tuple[dict, str]] = []

    for r in points:
        ck = text_key(r["embedding_text"], args.embedding_model)
        r["cache_key"] = ck
        if not args.force_refresh:
            hit = by_key.get(ck)
            if hit and isinstance(hit.get("embedding"), list) and hit["embedding"]:
                r["embedding"] = hit["embedding"]
                continue
        if args.cache_only:
            raise RuntimeError(
                f"--cache-only: missing embedding for {r['checkpoint_dir']} {r['lang']}"
            )
        to_embed.append((r, r["embedding_text"]))

    if to_embed:
        vecs = embed_openai([t[1] for t in to_embed], args.embedding_model)
        for (r, _), vec in zip(to_embed, vecs):
            r["embedding"] = vec

    for r in points:
        item = {
            "cache_key": r["cache_key"],
            "cache_schema_row": CACHE_SCHEMA,
            "checkpoint_dir": r["checkpoint_dir"],
            "lang": r["lang"],
            "jsonl_path": r["jsonl_path"],
            "jsonl_basename": r["jsonl_basename"],
            "instruction_id": r["instruction_id"],
            "instruction": r["instruction"],
            "response_used": r["response_used"],
            "response_index": r["response_index"],
            "embedding_text": r["embedding_text"],
            "text": r["embedding_text"],
            "embedding_model": args.embedding_model,
            "embedding": r["embedding"],
        }
        upsert(cache["items"], item)

    save_cache(cache_path, cache)
    print(f"Checkpoints×langs: {len(points)} rows → {cache_path}")


if __name__ == "__main__":
    main()
