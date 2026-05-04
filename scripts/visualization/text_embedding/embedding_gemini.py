import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from openai import OpenAI

try:
    from google.api_core import exceptions as google_api_exceptions
except ImportError:  # pragma: no cover
    google_api_exceptions = None

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


GEMINI_SCHEMA = "gemini_instruction_response_es_ru_first_v1"


def format_embedding_text(instruction: str, response: str) -> str:
    return f"### Instruction: {instruction} ### Response: {response}"


def text_key_embedding(embedding_text: str, embedding_model: str) -> str:
    return hashlib.sha256(f"{embedding_model}\n{embedding_text}".encode("utf-8")).hexdigest()


def text_key_generation(
    *,
    gemini_model: str,
    instruction: str,
    lang: str,
    source_key: str,
    jsonl_line: int,
) -> str:
    payload = f"{gemini_model}\n{lang}\n{source_key}\n{jsonl_line}\n{instruction}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "For first es + first ru row per lacomsa/icr/mapo JSONL: call Gemini to generate a reply, "
            "then embed (OpenAI) instruction + Gemini reply. Writes gemini-side cache JSON."
        )
    )
    p.add_argument(
        "--lacomsa-jsonl",
        default="/home/gangstat/NeMA_result/_pairs/train_lacomsa_relabeled.jsonl",
    )
    p.add_argument("--icr-jsonl", default="/home/gangstat/NeMA_result/_pairs/train_icr.jsonl")
    p.add_argument("--mapo-jsonl", default="/home/gangstat/NeMA_result/_pairs/train_mapo.jsonl")
    p.add_argument("--langs", default="es,ru", help="Comma-separated langs to grab first occurrence per file.")
    p.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help=(
            "Gemini model id (default: gemini-2.5-flash). Fallbacks if quota issues: gemini-1.5-flash, "
            "gemini-2.0-flash (free tier varies by model)."
        ),
    )
    p.add_argument(
        "--gemini-delay-sec",
        type=float,
        default=1.5,
        help="Sleep between Gemini calls to reduce rate-limit risk.",
    )
    p.add_argument(
        "--gemini-cache-path",
        default="/home/gangstat/NeMA_result/analysis/embeddings/gemini_same_instruction_es_ru_cache.json",
        help="Read/write Gemini + embedding cache.",
    )
    p.add_argument(
        "--sources",
        default="lacomsa,icr,mapo",
        help="Comma subset of lacomsa,icr,mapo. Each source adds first es+ru rows → 2 Gemini completions per source.",
    )
    p.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model.",
    )
    p.add_argument(
        "--cache-only",
        action="store_true",
        help="No Gemini/OpenAI unless an item is incomplete in cache.",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Redo Gemini generations and embeddings for every row.",
    )
    return p.parse_args()


def load_gemini_cache(path: Path):
    if not path.exists():
        return {
            "cache_schema": GEMINI_SCHEMA,
            "gemini_model": None,
            "embedding_model": None,
            "items": [],
        }
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("cache_schema", GEMINI_SCHEMA)
    data.setdefault("items", [])
    return data


def save_gemini_cache(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    data["cache_schema"] = GEMINI_SCHEMA
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def upsert(cache_items: list, item: dict):
    ck = item.get("gen_cache_key")
    if not ck:
        cache_items.append(item)
        return
    for i, old in enumerate(cache_items):
        if old.get("gen_cache_key") == ck:
            cache_items[i] = item
            return
    cache_items.append(item)


def collect_first_per_lang(jsonl_path: Path, target_langs: set):
    seen = set()
    out = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            lang = row.get("lang")
            if lang not in target_langs or lang in seen:
                continue
            seen.add(lang)
            out.append((line_no, row))
            if seen == target_langs:
                break
    missing = target_langs - seen
    if missing:
        raise ValueError(f"{jsonl_path}: no rows for langs {sorted(missing)}")
    return out


def select_jsonl_paths(keys, paths):
    bad = [k for k in keys if k not in paths]
    if bad:
        raise ValueError(f"Unknown --sources entry: {bad}. Use lacomsa, icr, mapo.")
    return {k: paths[k] for k in keys}


def build_gemini_rows(jsonl_paths: dict, target_langs: set):
    rows = []
    for source_key, path in jsonl_paths.items():
        path = Path(path)
        for line_no, row in collect_first_per_lang(path, target_langs):
            instruction = (row.get("instruction") or "").strip()
            if not instruction:
                raise ValueError(f"{path} line {line_no}: empty instruction")
            rows.append(
                {
                    "source_key": source_key,
                    "source_path": path.name,
                    "jsonl_line": line_no,
                    "lang": row.get("lang", ""),
                    "pair_id": row.get("id"),
                    "instruction": row.get("instruction") or "",
                }
            )
    return rows


def gemini_reply_for_instruction(*, instruction: str, lang: str, model_name: str) -> str:
    if genai is None:
        raise RuntimeError("Install google-generativeai: pip install google-generativeai")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = (
        f"The instruction below is from a preference dataset; the row language tag is '{lang}'. "
        "Follow the instruction and produce a single assistant response. "
        "Reply in the same language as the instruction when it is clearly Spanish or Russian; "
        "otherwise match the instruction language.\n\n"
        f"### Instruction\n{instruction}\n\n### Response\n"
    )
    try:
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.6, "max_output_tokens": 2048},
        )
    except Exception as exc:
        if google_api_exceptions and isinstance(exc, google_api_exceptions.ResourceExhausted):
            raise RuntimeError(
                f"Gemini quota/rate limit for model {model_name!r}. "
                "Try another model, e.g. --gemini-model gemini-1.5-flash. Original: "
                f"{exc}"
            ) from exc
        raise
    text = _extract_gemini_text(resp)
    if not str(text).strip():
        raise RuntimeError("Gemini returned empty text.")
    return text.strip()


def _extract_gemini_text(resp) -> str:
    if hasattr(resp, "text") and resp.text:
        return resp.text
    parts = []
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            t = getattr(part, "text", None)
            if t:
                parts.append(t)
    return "".join(parts)


def embed_openai(texts: list[str], model: str) -> list[list[float]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (needed for embeddings).")
    client = OpenAI(api_key=api_key)
    out = client.embeddings.create(model=model, input=texts)
    return [row.embedding for row in out.data]


def main():
    args = parse_args()
    target_langs = {x.strip() for x in args.langs.split(",") if x.strip()}
    if not target_langs:
        raise ValueError("--langs empty")

    all_paths = {
        "lacomsa": Path(args.lacomsa_jsonl),
        "icr": Path(args.icr_jsonl),
        "mapo": Path(args.mapo_jsonl),
    }
    source_keys = [x.strip() for x in args.sources.split(",") if x.strip()]
    if not source_keys:
        raise ValueError("--sources empty")
    jsonl_paths = select_jsonl_paths(source_keys, all_paths)
    cache_path = Path(args.gemini_cache_path)
    cache = load_gemini_cache(cache_path)

    if cache.get("gemini_model") not in (None, args.gemini_model) and not args.force_refresh:
        raise ValueError(
            f"Cache gemini_model {cache.get('gemini_model')!r} != {args.gemini_model!r}. "
            "Pass --force-refresh after switching models, or delete the Gemini cache JSON."
        )
    if cache.get("embedding_model") not in (None, args.embedding_model) and not args.force_refresh:
        raise ValueError(
            f"Cache embedding_model {cache.get('embedding_model')!r} != {args.embedding_model!r}"
        )

    cache["gemini_model"] = args.gemini_model
    cache["embedding_model"] = args.embedding_model

    by_gen = {it["gen_cache_key"]: it for it in cache["items"] if it.get("gen_cache_key")}

    rows = build_gemini_rows(jsonl_paths, target_langs)
    to_embed_batch: list[tuple[dict, str]] = []

    for row in rows:
        gck = text_key_generation(
            gemini_model=args.gemini_model,
            instruction=row["instruction"],
            lang=row["lang"],
            source_key=row["source_key"],
            jsonl_line=row["jsonl_line"],
        )
        row["gen_cache_key"] = gck
        existing = None if args.force_refresh else by_gen.get(gck)

        need_gemini = (
            args.force_refresh
            or not existing
            or not str(existing.get("gemini_response") or "").strip()
        )
        if need_gemini:
            if args.cache_only:
                raise RuntimeError(
                    f"--cache-only: missing Gemini text for {row['source_key']} {row['lang']} line {row['jsonl_line']}"
                )
            gemini_text = gemini_reply_for_instruction(
                instruction=row["instruction"],
                lang=row["lang"],
                model_name=args.gemini_model,
            )
            time.sleep(max(0.0, args.gemini_delay_sec))
        else:
            gemini_text = str(existing.get("gemini_response")).strip()

        et = format_embedding_text(row["instruction"], gemini_text)
        eck = text_key_embedding(et, args.embedding_model)
        row["gemini_response"] = gemini_text
        row["embedding_text"] = et
        row["cache_key"] = eck

        embedding_hit = (
            existing
            and not args.force_refresh
            and existing.get("cache_key") == eck
            and isinstance(existing.get("embedding"), list)
            and len(existing["embedding"]) > 0
        )
        if embedding_hit:
            row["embedding"] = existing["embedding"]
        else:
            if args.cache_only:
                raise RuntimeError(
                    f"--cache-only: missing embedding for {row['source_key']} {row['lang']}"
                )
            to_embed_batch.append((row, et))

    if to_embed_batch:
        vecs = embed_openai([t[1] for t in to_embed_batch], args.embedding_model)
        for (row, _), vec in zip(to_embed_batch, vecs):
            row["embedding"] = vec

    for row in rows:
        item = {
            "gen_cache_key": row["gen_cache_key"],
            "cache_key": row["cache_key"],
            "source_key": row["source_key"],
            "source_path": row["source_path"],
            "jsonl_line": row["jsonl_line"],
            "lang": row["lang"],
            "pair_id": row["pair_id"],
            "gemini_model": args.gemini_model,
            "embedding_model": args.embedding_model,
            "instruction": row["instruction"],
            "gemini_response": row["gemini_response"],
            "embedding_text": row["embedding_text"],
            "embedding": row["embedding"],
        }
        upsert(cache["items"], item)

    save_gemini_cache(cache_path, cache)
    print(f"Wrote {len(rows)} Gemini rows to {cache_path}")


if __name__ == "__main__":
    main()
