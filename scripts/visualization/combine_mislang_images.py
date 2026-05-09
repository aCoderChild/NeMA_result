#!/usr/bin/env python3
"""
Combine mislang_models and mislang_mismatch images by language.

This script reads images from two folders and combines them horizontally:
- analysis/figures/mislang_models_icr/mislang_models_{lang}.png
- analysis/figures/mislang_multilangs_icr/mislang_mismatch_{lang}.png

Output: analysis/figures/combined_mislang_models_icr/combined_mislang_{lang}.png
"""

import argparse
import os
from pathlib import Path

from PIL import Image


LANGS = ["de", "en", "es", "fr", "ru"]


def combine_images(
    models_dir: str,
    mismatch_dir: str,
    output_dir: str,
    lang: str,
) -> None:
    """
    Combine mislang_models and mislang_mismatch images for a given language.
    
    Args:
        models_dir: Path to mislang_models_icr folder
        mismatch_dir: Path to mislang_multilangs_icr folder
        output_dir: Path to output combined folder
        lang: Language code (de, en, es, fr, ru)
    """
    models_path = os.path.join(models_dir, f"mislang_models_{lang}.png")
    mismatch_path = os.path.join(mismatch_dir, f"mislang_mismatch_{lang}.png")

    if not os.path.exists(models_path):
        print(f"Warning: {models_path} not found, skipping {lang}")
        return

    if not os.path.exists(mismatch_path):
        print(f"Warning: {mismatch_path} not found, skipping {lang}")
        return

    # Load images
    models_img = Image.open(models_path)
    mismatch_img = Image.open(mismatch_path)

    # Get dimensions
    models_width, models_height = models_img.size
    mismatch_width, mismatch_height = mismatch_img.size

    # Resize mismatch image to match height of models image if different
    if models_height != mismatch_height:
        ratio = models_height / mismatch_height
        new_width = int(mismatch_width * ratio)
        mismatch_img = mismatch_img.resize((new_width, models_height), Image.Resampling.LANCZOS)
        mismatch_width, mismatch_height = mismatch_img.size

    # Create new image with combined width
    combined_width = models_width + mismatch_width
    combined_height = models_height
    combined_img = Image.new("RGB", (combined_width, combined_height), color="white")

    # Paste images side by side
    combined_img.paste(models_img, (0, 0))
    combined_img.paste(mismatch_img, (models_width, 0))

    # Save combined image
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"combined_mislang_{lang}.png")
    combined_img.save(output_path, dpi=(300, 300))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine mislang_models and mislang_mismatch images by language"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="analysis/figures/mislang_models_lacomsa",
        help="Path to mislang_models_icr folder",
    )
    parser.add_argument(
        "--mismatch-dir",
        type=str,
        default="analysis/figures/mislang_multilangs_lacomsa",
        help="Path to mislang_multilangs_icr folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/figures/combined_mislang_models_lacomsa",
        help="Path to output combined folder",
    )
    args = parser.parse_args()

    if not os.path.exists(args.models_dir):
        print(f"Error: Models directory not found: {args.models_dir}")
        return

    if not os.path.exists(args.mismatch_dir):
        print(f"Error: Mismatch directory not found: {args.mismatch_dir}")
        return

    print(f"Combining images from {args.models_dir} and {args.mismatch_dir}...")
    for lang in LANGS:
        combine_images(args.models_dir, args.mismatch_dir, args.output_dir, lang)

    print("Done!")


if __name__ == "__main__":
    main()
