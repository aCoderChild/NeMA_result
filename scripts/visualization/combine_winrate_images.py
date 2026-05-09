#!/usr/bin/env python3
import argparse
import os

from PIL import Image


DEFAULT_INPUTS = [
    "visualisation/winrate_per_method_icr.png",
    "visualisation/winrate_per_method_lacomsa.png",
    "visualisation/winrate_per_method_mapo.png",
]
DEFAULT_OUTPUT = "visualisation/combined_winrate_per_method.png"


def combine_horizontally(image_paths, output_path, bg_color=(255, 255, 255)):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        target_height = max(img.height for img in images)
        resized_images = []
        total_width = 0

        for img in images:
            if img.height != target_height:
                ratio = target_height / img.height
                new_width = int(round(img.width * ratio))
                img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
            resized_images.append(img)
            total_width += img.width

        canvas = Image.new("RGB", (total_width, target_height), color=bg_color)

        x = 0
        for img in resized_images:
            canvas.paste(img, (x, 0))
            x += img.width

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        canvas.save(output_path)
        print(f"Saved combined image: {output_path}")
    finally:
        for img in images:
            img.close()


def main():
    parser = argparse.ArgumentParser(description="Combine three win-rate PNGs horizontally.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for path in args.inputs:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input image: {path}")

    combine_horizontally(args.inputs, args.output)


if __name__ == "__main__":
    main()