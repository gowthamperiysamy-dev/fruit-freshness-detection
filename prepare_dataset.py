
import random
import shutil
from pathlib import Path

SOURCE = Path("Original Image")
DESTINATION = Path("fruit_dataset")

random.seed(42)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not SOURCE.exists():
    raise FileNotFoundError("Original Image folder not found.")

for split in ["train", "val", "test"]:
    for category in ["Fresh", "Rotten"]:
        (DESTINATION / split / category).mkdir(
            parents=True,
            exist_ok=True
        )

for category in ["Fresh", "Rotten"]:

    images = []

    for folder in SOURCE.iterdir():
        if folder.is_dir() and folder.name.startswith(category):
            images.extend([
                file for file in folder.rglob("*")
                if file.suffix.lower() in extensions
            ])

    random.shuffle(images)

    total = len(images)

    if total == 0:
        raise ValueError(f"No images found for {category}")

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    print(f"\n{category}: {total} images")

    for split, files in splits.items():

        target = DESTINATION / split / category

        for index, file in enumerate(files):
            shutil.copy2(
                file,
                target / f"{category}_{index}{file.suffix.lower()}"
            )

        print(f"{split}: {len(files)} images")

print("\nDataset preparation completed!")
