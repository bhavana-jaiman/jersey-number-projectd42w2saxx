import os
import shutil
import random
import argparse


def split_dataset(dataset_dir, train_ratio=0.8, seed=42):

    dataset_dir = os.path.abspath(dataset_dir)

    if not os.path.isdir(dataset_dir):
        print(f"ERROR: Directory not found: {dataset_dir}")
        return

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    image_extensions = (".jpg", ".jpeg", ".png")

    # Get images directly inside dataset directory
    images = [
        f for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f))
        and f.lower().endswith(image_extensions)
    ]

    if len(images) == 0:
        print("ERROR: No images found directly inside dataset directory.")
        return

    # Reproducible random split
    random.seed(seed)
    random.shuffle(images)

    train_count = int(len(images) * train_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:]

    print(f"Total images      : {len(images)}")
    print(f"Training images   : {len(train_images)}")
    print(f"Validation images : {len(val_images)}")

    # Copy images and corresponding labels
    def copy_data(image_list, destination):

        copied_images = 0
        copied_labels = 0

        for image_name in image_list:

            image_source = os.path.join(dataset_dir, image_name)
            image_destination = os.path.join(destination, image_name)

            shutil.copy2(image_source, image_destination)
            copied_images += 1

            # Find corresponding TXT label
            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_source = os.path.join(dataset_dir, label_name)

            if os.path.exists(label_source):

                label_destination = os.path.join(
                    destination,
                    label_name
                )

                shutil.copy2(label_source, label_destination)
                copied_labels += 1

            else:
                print(f"WARNING: Label not found for {image_name}")

        return copied_images, copied_labels

    train_img, train_lbl = copy_data(train_images, train_dir)
    val_img, val_lbl = copy_data(val_images, val_dir)

    print()
    print("====================================")
    print("Dataset split completed")
    print("====================================")
    print(f"Train images       : {train_img}")
    print(f"Train labels       : {train_lbl}")
    print(f"Validation images  : {val_img}")
    print(f"Validation labels  : {val_lbl}")
    print()
    print(f"Train directory    : {train_dir}")
    print(f"Validation directory: {val_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Split dataset into train and validation"
    )

    parser.add_argument(
        "dataset_dir",
        help="Dataset directory, e.g. demo/simple2D"
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training ratio (default: 0.8)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    split_dataset(
        args.dataset_dir,
        args.train_ratio,
        args.seed
    )
