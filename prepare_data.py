import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
TRAIN_IMG_DIR = WORKSPACE / "train" / "images"
TRAIN_LBL_DIR = WORKSPACE / "train" / "labels"
VAL_IMG_DIR = WORKSPACE / "val" / "images"
VAL_LBL_DIR = WORKSPACE / "val" / "labels"
DATA_YAML = WORKSPACE / "data.yaml"

SEED = 42
VAL_SAMPLE_COUNT = 3
TARGET_TRAIN_COUNT = 500
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_image_pairs():
    pairs = []
    for image_path in sorted(TRAIN_IMG_DIR.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = TRAIN_LBL_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Khong tim thay nhan cho anh: {image_path.name}")
        pairs.append((image_path, label_path))
    return pairs


def move_validation_samples(pairs):
    VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    VAL_LBL_DIR.mkdir(parents=True, exist_ok=True)

    existing_val_images = [
        path for path in VAL_IMG_DIR.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(existing_val_images) >= VAL_SAMPLE_COUNT:
        print(f"--> Da co {len(existing_val_images)} anh trong val/, bo qua tach lai.")
        return

    random_generator = random.Random(SEED)
    selected_pairs = random_generator.sample(pairs, VAL_SAMPLE_COUNT)
    for image_path, label_path in selected_pairs:
        shutil.move(str(image_path), str(VAL_IMG_DIR / image_path.name))
        shutil.move(str(label_path), str(VAL_LBL_DIR / label_path.name))
        print(f"[VAL] {image_path.name}")


def update_data_yaml():
    if not DATA_YAML.exists():
        return

    lines = DATA_YAML.read_text(encoding="utf-8").splitlines(keepends=True)
    updated_lines = []
    for line in lines:
        if line.lstrip().startswith("train:"):
            updated_lines.append("train: train/images\n")
        elif line.lstrip().startswith("val:"):
            updated_lines.append("val: val/images\n")
        else:
            updated_lines.append(line)
    DATA_YAML.write_text("".join(updated_lines), encoding="utf-8")


def run_augmentation(target_count):
    if target_count <= 0:
        print("--> Train da co du 500 anh, khong can augment.")
        return

    environment = os.environ.copy()
    environment["AUGMENT_TARGET_COUNT"] = str(target_count)
    print(f"--> Chay augment.py de tao them {target_count} anh...")
    subprocess.run(
        [sys.executable, str(WORKSPACE / "augment.py")],
        cwd=str(WORKSPACE),
        env=environment,
        check=True,
    )


def main():
    print("=" * 60)
    print("CHUAN BI DU LIEU TRAIN / VAL")
    print("=" * 60)

    pairs = get_image_pairs()
    if len(pairs) < VAL_SAMPLE_COUNT:
        raise ValueError(f"Can it nhat {VAL_SAMPLE_COUNT} cap anh-nhan trong train/.")

    move_validation_samples(pairs)
    update_data_yaml()

    train_images_before_augmentation = len(
        [path for path in TRAIN_IMG_DIR.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    )
    target_count = TARGET_TRAIN_COUNT - train_images_before_augmentation
    run_augmentation(target_count)

    train_images = [
        path for path in TRAIN_IMG_DIR.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    train_labels = list(TRAIN_LBL_DIR.glob("*.txt"))
    val_images = [
        path for path in VAL_IMG_DIR.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    val_labels = list(VAL_LBL_DIR.glob("*.txt"))

    print("\n" + "=" * 60)
    print(f"Train: {len(train_images)} anh, {len(train_labels)} nhan")
    print(f"Val:   {len(val_images)} anh, {len(val_labels)} nhan")
    if len(train_images) != TARGET_TRAIN_COUNT:
        raise RuntimeError(f"Train chua dat {TARGET_TRAIN_COUNT} anh.")
    if len(train_images) != len(train_labels) or len(val_images) != len(val_labels):
        raise RuntimeError("Phat hien anh va nhan khong dong bo.")
    print("HOAN THANH: du lieu hop le va dong bo.")


if __name__ == "__main__":
    main()