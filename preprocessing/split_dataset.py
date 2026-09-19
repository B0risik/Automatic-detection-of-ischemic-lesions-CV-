"""
Разделение датасета на train/val по пациентам (без утечки данных)
и создание финальной структуры для YOLOv8.
"""
import shutil
import random
from pathlib import Path

# ==== НАСТРОЙКИ ====
SRC_DIR   = Path(r"/yolo_dataset_raw")
DST_DIR   = Path(r"/yolo_dataset")
VAL_RATIO = 0.2     # 20% пациентов — в валидацию
SEED      = 42      # для воспроизводимости

random.seed(SEED)


def main():
    # Собираем список пациентов
    patients = sorted([p.name for p in (SRC_DIR / "images").iterdir() if p.is_dir()])
    print(f"Всего пациентов: {len(patients)}")

    # Перемешиваем и делим
    random.shuffle(patients)
    n_val = max(1, int(len(patients) * VAL_RATIO))
    val_patients   = set(patients[:n_val])
    train_patients = set(patients[n_val:])
    print(f"Train: {len(train_patients)} пациентов")
    print(f"Val:   {len(val_patients)} пациентов")

    # Создаём структуру
    for split in ["train", "val"]:
        (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Копируем файлы
    counts = {"train": 0, "val": 0}
    for split, pat_set in [("train", train_patients), ("val", val_patients)]:
        for pid in pat_set:
            # Изображения
            img_src_dir = SRC_DIR / "images" / pid
            for img in img_src_dir.glob("*.png"):
                shutil.copy(img, DST_DIR / "images" / split / img.name)

            # Аннотации
            lbl_src_dir = SRC_DIR / "labels_yolo" / pid
            for lbl in lbl_src_dir.glob("*.txt"):
                shutil.copy(lbl, DST_DIR / "labels" / split / lbl.name)

            counts[split] += len(list(img_src_dir.glob("*.png")))

    print(f"\nГотово!")
    print(f"   Train: {counts['train']} изображений")
    print(f"   Val:   {counts['val']} изображений")
    print(f"   Папка: {DST_DIR}")

    # Сохраняем список пациентов (для воспроизводимости)
    (DST_DIR / "train_patients.txt").write_text("\n".join(sorted(train_patients)))
    (DST_DIR / "val_patients.txt").write_text("\n".join(sorted(val_patients)))


if __name__ == "__main__":
    main()