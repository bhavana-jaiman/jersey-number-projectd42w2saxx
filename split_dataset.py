import os
import shutil
import random
import argparse


def split_dataset(dataset_dir, val_ratio=0.2, seed=42, remove_source=False):

    dataset_dir = os.path.abspath(dataset_dir)

    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    dataset_name = os.path.basename(dataset_dir.rstrip("/"))

    parent_dir = os.path.dirname(dataset_dir)

    train_dir = os.path.join(parent_dir, dataset_name + "_train")
    val_dir = os.path.join(parent_dir, dataset_name + "_validation")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Collect image-label pairs recursively
    # ---------------------------------------------------------
    pairs = []

    for root, dirs, files in os.walk(dataset_dir):

        for file in files:

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(root, file)

            base_name = os.path.splitext(file)[0]

            label_path = None

            # Look for matching label file
            for ext in [".txt"]:
                possible_label = os.path.join(root, base_name + ext)

                if os.path.isfile(possible_label):
                    label_path = possible_label
                    break

            if label_path is not None:
                pairs.append((image_path, label_path))

    print(f"\nDataset: {dataset_name}")
    print(f"Total image-label pairs found: {len(pairs)}")

    if len(pairs) == 0:
        print("ERROR: No image-label pairs found.")
        return

    # ---------------------------------------------------------
    # Shuffle
    # ---------------------------------------------------------
    random.seed(seed)
    random.shuffle(pairs)

    # ---------------------------------------------------------
    # Train / Validation split
    # ---------------------------------------------------------
    val_count = int(len(pairs) * val_ratio)

    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    print(f"Training pairs:   {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")

    # ---------------------------------------------------------
    # Copy files directly into train/validation directories
    # ---------------------------------------------------------

    def copy_pairs(pair_list, destination):

        for image_path, label_path in pair_list:

            image_name = os.path.basename(image_path)
            label_name = os.path.basename(label_path)

            shutil.copy2(
                image_path,
                os.path.join(destination, image_name)
            )

            shutil.copy2(
                label_path,
                os.path.join(destination, label_name)
            )

    copy_pairs(train_pairs, train_dir)
    copy_pairs(val_pairs, val_dir)

    print("\nFiles copied successfully.")

    print(f"\nTraining directory:")
    print(train_dir)

    print(f"\nValidation directory:")
    print(val_dir)

    # ---------------------------------------------------------
    # OPTIONAL: Remove original numbered subfolders
    # ---------------------------------------------------------

    if remove_source:

        print("\nRemoving original subfolders...")

        for item in os.listdir(dataset_dir):

            item_path = os.path.join(dataset_dir, item)

            if os.path.isdir(item_path):

                shutil.rmtree(item_path)

                print(f"Removed: {item_path}")

        print("\nOriginal subfolders removed.")

    print("\n========== DONE ==========\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Split Simple2D/Complex2D dataset into flat train and validation directories."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory, e.g. simple2D or complex2D"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio. Default = 0.2"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default = 42"
    )

    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Remove original numbered subfolders after successful copying"
    )

    args = parser.parse_args()

    split_dataset(
        dataset_dir=args.dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        remove_source=args.remove_source
    )
