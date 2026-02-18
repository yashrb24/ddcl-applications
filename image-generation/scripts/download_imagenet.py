"""
Download ImageNet-1k 256×256 from HuggingFace and save as ImageFolder layout.

Source: https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256

Output structure (human-readable class names):
  <output_dir>/train/tench/00000.png
  <output_dir>/train/goldfish/00001.png
  <output_dir>/val/tench/00000.png
  ...

This format works directly with our VQGanVAETrainer (--image_folder) and
torchvision.datasets.ImageFolder (Stage 2 MaskGit training).

Usage:
  python scripts/download_imagenet.py --output_dir ./data/imagenet-256
  python scripts/download_imagenet.py --output_dir ./data/imagenet-256 --splits train val

Requires:
  pip install datasets pillow
"""

import argparse
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download ImageNet-1k 256×256")
    parser.add_argument("--output_dir", type=str, default="./data/imagenet-256")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"],
                        help="Which splits to download (use 'val' for validation)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Map split names (HF uses "validation", ImageFolder convention uses "val")
    split_dir_names = {"train": "train", "validation": "val"}

    for split in args.splits:
        hf_split = split if split != "val" else "validation"
        dir_name = split_dir_names.get(hf_split, hf_split)

        print(f"Loading split '{hf_split}' from HuggingFace...")
        ds = load_dataset(
            "evanarlian/imagenet_1k_resized_256",
            split=hf_split,
            trust_remote_code=True,
        )

        # Get label -> human-readable class name mapping
        label_names = ds.features["label"].names
        print(f"  {len(ds)} images, {len(label_names)} classes")

        # Track per-class counters for unique filenames
        counters = defaultdict(int)

        split_dir = output_dir / dir_name

        print(f"Saving to {split_dir}/...")
        for idx, example in enumerate(ds):
            label = example["label"]
            class_name = label_names[label]

            # sanitize class name for filesystem (replace spaces/slashes)
            class_name = class_name.replace("/", "_").replace(" ", "_")

            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            count = counters[class_name]
            counters[class_name] += 1

            img_path = class_dir / f"{count:05d}.png"
            example["image"].save(str(img_path))

            if idx % 10000 == 0:
                print(f"  [{dir_name}] {idx}/{len(ds)} saved...")

        print(f"  Done: {len(ds)} images saved to {split_dir}/")

    print(f"\nAll done. Dataset at: {output_dir}")
    print(f"Use --image_folder {output_dir}/train for Stage 1 training.")


if __name__ == "__main__":
    main()
