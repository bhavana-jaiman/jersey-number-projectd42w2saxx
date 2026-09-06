import os
import shutil
import argparse


def flatten_dataset(dataset_dir):
    dataset_dir = os.path.abspath(dataset_dir)

    if not os.path.isdir(dataset_dir):
        print(f"ERROR: Directory not found: {dataset_dir}")
        return

    # Class folders such as 0, 1, 2, ..., 99
    class_dirs = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    print(f"Dataset: {dataset_dir}")
    print(f"Found {len(class_dirs)} subfolders")

    image_extensions = (".jpg", ".jpeg", ".png")

    copied_images = 0
    copied_labels = 0

    for class_name in sorted(class_dirs, key=lambda x: x):
        class_dir = os.path.join(dataset_dir, class_name)

        for filename in os.listdir(class_dir):
            source = os.path.join(class_dir, filename)

            if not os.path.isfile(source):
                continue

            # Process images
            if filename.lower().endswith(image_extensions):

                destination = os.path.join(dataset_dir, filename)

                # Avoid overwriting if the same filename exists
                if os.path.exists(destination):
                    name, ext = os.path.splitext(filename)
                    destination = os.path.join(
                        dataset_dir,
                        f"{class_name}_{name}{ext}"
                    )

                shutil.copy2(source, destination)
                copied_images += 1

                # Copy corresponding TXT label
                label_name = os.path.splitext(filename)[0] + ".txt"
                label_source = os.path.join(class_dir, label_name)

                if os.path.exists(label_source):
                    label_destination = os.path.join(
                        dataset_dir,
                        os.path.basename(destination).rsplit(".", 1)[0] + ".txt"
                    )

                    shutil.copy2(label_source, label_destination)
                    copied_labels += 1

    print()
    print("================================")
    print("Dataset conversion completed")
    print("================================")
    print(f"Images copied : {copied_images}")
    print(f"Labels copied : {copied_labels}")
    print(f"Output folder : {dataset_dir}")
    print()
    print("You can now check with:")
    print(f"ls {dataset_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Flatten jersey dataset into one directory"
    )

    parser.add_argument(
        "dataset_dir",
        help="Path to dataset, e.g. demo/simple2D"
    )

    args = parser.parse_args()

    flatten_dataset(args.dataset_dir)
