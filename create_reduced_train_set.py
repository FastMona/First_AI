"""Create a reduced MNIST training set with one underrepresented digit.

Reads MNIST IDX files from training_data/MNIST/raw by default (or a user-provided
raw folder), prompts for a digit to underrepresent, and keeps only 0-100 images
for that digit while keeping all other training images.

Output is written in MNIST-compatible IDX format to reduced_train_set/MNIST/raw,
including copied test files.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import numpy as np


def read_idx_labels(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"Labels file too small: {path}")
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"Unexpected labels magic number {magic} in {path}")
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    if labels.size != count:
        raise ValueError(
            f"Label count mismatch in {path}: header={count}, parsed={labels.size}"
        )
    return labels.copy()


def read_idx_images(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError(f"Images file too small: {path}")
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"Unexpected images magic number {magic} in {path}")

    image_size = rows * cols
    expected = count * image_size
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    if images.size != expected:
        raise ValueError(
            f"Image count mismatch in {path}: expected_pixels={expected}, parsed={images.size}"
        )
    return images.reshape(count, rows, cols).copy()


def write_idx_labels(path: Path, labels: np.ndarray) -> None:
    header = struct.pack(">II", 2049, int(labels.shape[0]))
    path.write_bytes(header + labels.astype(np.uint8).tobytes())


def write_idx_images(path: Path, images: np.ndarray) -> None:
    if images.ndim != 3:
        raise ValueError("Images must be a 3D array: (count, rows, cols)")
    count, rows, cols = images.shape
    header = struct.pack(">IIII", 2051, int(count), int(rows), int(cols))
    path.write_bytes(header + images.astype(np.uint8).tobytes())


def ask_underrepresented_digit() -> int:
    while True:
        raw = input("Enter digit to underrepresent (0-9): ").strip()
        try:
            digit = int(raw)
        except ValueError:
            print("Please enter a whole number from 0 to 9.")
            continue
        if 0 <= digit <= 9:
            return digit
        print("Digit must be between 0 and 9.")


def ask_kept_count() -> int:
    while True:
        raw = input("How many images of that digit should be kept (0-100)? ").strip()
        try:
            count = int(raw)
        except ValueError:
            print("Please enter a whole number from 0 to 100.")
            continue
        if 0 <= count <= 100:
            return count
        print("Count must be between 0 and 100.")


def build_reduced_train_set(raw_input_dir: Path, output_root: Path) -> None:
    train_images_path = raw_input_dir / "train-images-idx3-ubyte"
    train_labels_path = raw_input_dir / "train-labels-idx1-ubyte"
    test_images_path = raw_input_dir / "t10k-images-idx3-ubyte"
    test_labels_path = raw_input_dir / "t10k-labels-idx1-ubyte"

    required = [
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required MNIST raw files:\n" + "\n".join(missing))

    print("\nLoading training IDX files...")
    train_images = read_idx_images(train_images_path)
    train_labels = read_idx_labels(train_labels_path)

    if train_images.shape[0] != train_labels.shape[0]:
        raise ValueError(
            "Training images/labels length mismatch: "
            f"{train_images.shape[0]} vs {train_labels.shape[0]}"
        )

    under_digit = ask_underrepresented_digit()
    kept_count = ask_kept_count()

    target_indices = np.where(train_labels == under_digit)[0]
    keep_target_indices = target_indices[:kept_count]
    keep_other_indices = np.where(train_labels != under_digit)[0]
    keep_indices = np.concatenate([keep_other_indices, keep_target_indices])

    # Keep deterministic order by original sample index.
    keep_indices.sort()

    reduced_images = train_images[keep_indices]
    reduced_labels = train_labels[keep_indices]

    output_raw_dir = output_root / "MNIST" / "raw"
    output_raw_dir.mkdir(parents=True, exist_ok=True)

    print("\nWriting reduced training IDX files...")
    write_idx_images(output_raw_dir / "train-images-idx3-ubyte", reduced_images)
    write_idx_labels(output_raw_dir / "train-labels-idx1-ubyte", reduced_labels)

    # Copy test set unchanged so torchvision MNIST works with this root.
    shutil.copy2(test_images_path, output_raw_dir / "t10k-images-idx3-ubyte")
    shutil.copy2(test_labels_path, output_raw_dir / "t10k-labels-idx1-ubyte")

    per_digit_counts = {d: int((reduced_labels == d).sum()) for d in range(10)}
    print("\nReduced training set created successfully.")
    print(f"Output folder: {output_root}")
    print(f"Total training images: {reduced_labels.shape[0]}")
    print(f"Underrepresented digit: {under_digit}")
    print(f"Kept images for digit {under_digit}: {per_digit_counts[under_digit]}")
    print("Per-digit training counts:")
    for d in range(10):
        print(f"  {d}: {per_digit_counts[d]}")


def main() -> None:
    default_input = Path(r"C:\Users\fastm\Documents_local\Repositories_local\Python_stuff\First_AI\training_data\MNIST\raw")
    output_root = Path("reduced_train_set")

    user_input = input(
        "Raw MNIST input folder "
        f"[{default_input}]: "
    ).strip()
    raw_input_dir = Path(user_input) if user_input else default_input

    print("\nThis will create a reduced MNIST training dataset in 'reduced_train_set'.")
    if output_root.exists():
        overwrite = input(
            f"Output folder '{output_root}' already exists. Overwrite it? (yes/no): "
        ).strip().lower()
        if overwrite not in {"yes", "y"}:
            print("Cancelled.")
            return
        shutil.rmtree(output_root)

    build_reduced_train_set(raw_input_dir=raw_input_dir, output_root=output_root)


if __name__ == "__main__":
    main()
