import argparse
import os

from PIL import Image


DEFAULT_INPUTS = [
    "/home/gangstat/NeMA_result/analysis/figures/overview_icr.png",
    "/home/gangstat/NeMA_result/analysis/figures/overview_lacomsa.png",
    "/home/gangstat/NeMA_result/analysis/figures/overview_mapo.png",
]
DEFAULT_OUTPUT = "/home/gangstat/NeMA_result/analysis/figures/overview_stacked_vertical.png"


def stack_vertical(image_paths, output_path, bg_color=(255, 255, 255)):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)

    canvas = Image.new("RGB", (max_width, total_height), color=bg_color)

    y = 0
    for img in images:
        x = (max_width - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print(f"Saved stacked image: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stack overview PNGs vertically.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for p in args.inputs:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input image: {p}")

    stack_vertical(args.inputs, args.output)


if __name__ == "__main__":
    main()
