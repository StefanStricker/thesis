import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "StefanStricker/trashvariety"

OUT_ROOT = Path("data/datasets/trashvariety")
CACHE_DIR = Path("data/hf_cache/trashvariety_repo")
REPO_IMAGES_DIR = Path("self-collected")

def main():
    repo_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(CACHE_DIR),
            allow_patterns=[str(REPO_IMAGES_DIR / "**")],
        )
    )

    src_root = repo_dir / REPO_IMAGES_DIR
    if not src_root.exists():
        raise FileNotFoundError(src_root)

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(src_root.iterdir()):
        if not class_dir.is_dir():
            continue

        has_images = any(p.is_file() and p.suffix.lower() == ".jpg" for p in class_dir.iterdir())
        if not has_images:
            continue

        dst_class_dir = OUT_ROOT / class_dir.name
        shutil.copytree(class_dir, dst_class_dir)

        num_imgs = sum(1 for p in dst_class_dir.iterdir() if p.is_file())
        print(f"Copied {class_dir.name} -> {dst_class_dir} ({num_imgs} images)")

    print("\nDownload complete")

if __name__ == "__main__":
    main()
