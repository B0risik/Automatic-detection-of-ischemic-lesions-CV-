"""
Конвертация ISLES в формат, готовый для YOLOv8.
Создаёт 2-канальные PNG (R=DWI, G=ADC) + PNG-маски с апскейлом.
"""
import os
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ==== НАСТРОЙКИ ====
DATASET_ROOT = Path(r"/isles2022/ISLES/dataset")
RAW_DIR      = DATASET_ROOT / "rawdata"
DERIV_DIR    = DATASET_ROOT / "derivatives"
OUTPUT_DIR   = Path(r"/yolo_dataset_raw")

# Размер, к которому приводим все изображения (YOLOv8 обучим на 640, но храним 256)
TARGET_SIZE  = 256

n = 10

# Ограничение по числу пациентов (None)
MAX_PATIENTS = None   # n для теста, потом None

# Негативные срезы (без очага) сохраняем каждый N-й
SAVE_EMPTY_EVERY = 5

# Формат файлов: '.nii.gz' или '.nii' — определится автоматически
def find_file(folder, patient_id, suffix):
    """Ищет файл с расширением .nii.gz или .nii."""
    for ext in [".nii.gz", ".nii"]:
        p = folder / patient_id / "ses-0001" / f"{patient_id}_ses-0001_{suffix}{ext}"
        if p.exists():
            return p
    return None


def normalize_mri(volume: np.ndarray) -> np.ndarray:
    """Нормализация к 0..255 с percentile clipping."""
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    if p99 - p1 < 1e-6:
        return np.zeros_like(volume, dtype=np.uint8)
    volume = (volume - p1) / (p99 - p1) * 255.0
    return volume.astype(np.uint8)


def process_patient(patient_id: str):
    """Обработка одного пациента."""
    dwi_path = find_file(RAW_DIR, patient_id, "dwi")
    adc_path = find_file(RAW_DIR, patient_id, "adc")
    msk_path = find_file(DERIV_DIR, patient_id, "msk")

    if not (dwi_path and adc_path and msk_path):
        print(f"⚠️  Пропуск {patient_id}: не все файлы найдены")
        return 0

    dwi = nib.load(str(dwi_path)).get_fdata()
    adc = nib.load(str(adc_path)).get_fdata()
    msk = nib.load(str(msk_path)).get_fdata()

    # Транспонируем: обычно у NIfTI оси (X, Y, Z) — нам нужны (Z, X, Y) для срезов
    # Проверим на всякий случай: количество срезов — минимальная ось
    n_slices = dwi.shape[2]
    assert dwi.shape == adc.shape == msk.shape, f"Размерности не совпадают у {patient_id}"

    img_dir = OUTPUT_DIR / "images" / patient_id
    lbl_dir = OUTPUT_DIR / "labels" / patient_id
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i in range(n_slices):
        slice_dwi = dwi[:, :, i]
        slice_adc = adc[:, :, i]
        slice_msk = msk[:, :, i]

        # Пропускаем полностью пустые срезы
        if slice_dwi.max() < 1e-3 and slice_adc.max() < 1e-3:
            continue

        has_lesion = slice_msk.max() > 0
        if not has_lesion and (i % SAVE_EMPTY_EVERY != 0):
            continue

        # Нормализация
        dwi_norm = normalize_mri(slice_dwi)
        adc_norm = normalize_mri(slice_adc)
        msk_bin  = (slice_msk > 0).astype(np.uint8) * 255

        # Ориентация: поворот на 90°
        dwi_norm = np.rot90(dwi_norm)
        adc_norm = np.rot90(adc_norm)
        msk_bin  = np.rot90(msk_bin)

        # Апскейл до TARGET_SIZE × TARGET_SIZE
        dwi_img = Image.fromarray(dwi_norm).resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
        adc_img = Image.fromarray(adc_norm).resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
        msk_img = Image.fromarray(msk_bin).resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)

        # 2-канальное RGB: R=DWI, G=ADC
        dwi_arr = np.array(dwi_img)
        adc_arr = np.array(adc_img)
        rgb = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
        rgb[..., 0] = dwi_arr
        rgb[..., 1] = adc_arr

        fname = f"{patient_id}_slice{i:03d}"
        Image.fromarray(rgb).save(img_dir / f"{fname}.png")
        msk_img.save(lbl_dir / f"{fname}.png")
        saved += 1

    return saved


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patient_ids = sorted([p.name for p in RAW_DIR.iterdir() if p.is_dir()])
    if MAX_PATIENTS is not None:
        patient_ids = patient_ids[:MAX_PATIENTS]

    print(f"Обрабатываю {len(patient_ids)} пациентов...")
    total = 0
    for pid in tqdm(patient_ids):
        total += process_patient(pid)

    print(f"\nСохранено {total} срезов в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()