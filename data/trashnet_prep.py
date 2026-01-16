import shutil
import random
from pathlib import Path

SEED = 16
rng = random.Random(SEED)

SPLITS = (0.8, 0.1, 0.1)

RAW = Path("data/TrashNet_pre")
DST = Path("data/datasets/trashnet")          


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir())

def copy_split(img_paths, split_name: str, class_name: str):
    out_dir = DST / split_name / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in img_paths:
        shutil.copy2(p, out_dir / p.name)


def split_class(images):
    images = list(images)
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * SPLITS[0])
    n_val = int(n * SPLITS[1])

    train = images[:n_train]
    val = images[n_train : n_train + n_val]
    test = images[n_train + n_val :]

    return train, val, test

if __name__ == "__main__":
    if not RAW.exists():
        raise FileNotFoundError(f"RAW folder not found: {RAW.resolve()}")

    class_folders = [p for p in sorted(RAW.iterdir()) if p.is_dir()]
    if not class_folders:
        raise RuntimeError(f"No class subfolders found under: {RAW.resolve()}")

    for class_folder in class_folders:
        class_name = class_folder.name
        images = list_images(class_folder)
        if not images:
            print(f"Skipping empty class folder: {class_name}")
            continue

        train, val, test = split_class(images)

        copy_split(train, "train", class_name)
        copy_split(val, "val", class_name)
        copy_split(test, "test", class_name)

        print(f"{class_name:>10}: total={len(images)}  train={len(train)}  val={len(val)}  test={len(test)}")