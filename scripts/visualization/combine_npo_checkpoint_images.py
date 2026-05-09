#!/usr/bin/env python3
import argparse
import os

from PIL import Image


DEFAULT_INPUTS = [
    "visualisation/winrate_improvement_npo_checkpoints[1, 10].png",
    "visualisation/lc_winrate_improvement_npo_checkpoints[1, 10].png",
]
DEFAULT_OUTPUT = "visualisation/combined_npo_checkpoint_improvements[1, 10].png"


def combine_horizontally(image_paths, output_path, bg_color=(255, 255, 255)):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    combined_width = sum(img.width for img in images)
    combined_height = max(img.height for img in images)

    canvas = Image.new("RGB", (combined_width, combined_height), color=bg_color)

    x = 0
    for img in images:
        y = (combined_height - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print(f"Saved combined image: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine the NPO checkpoint improvement charts horizontally."
    )
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for path in args.inputs:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input image: {path}")

    combine_horizontally(args.inputs, args.output)


if __name__ == "__main__":
    main()