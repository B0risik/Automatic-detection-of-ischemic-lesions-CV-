"""
Конвертация PNG-масок в YOLO-аннотации (полигоны).
Для каждой маски создаётся .txt с координатами контуров.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==== НАСТРОЙКИ ====
DATASET_DIR = Path(r"/yolo_dataset_raw")
LABELS_DIR  = DATASET_DIR / "labels"        # PNG-маски (вход)
YOLO_DIR    = DATASET_DIR / "labels_yolo"   # TXT-аннотации (выход)

# Минимальная площадь контура в пикселях (отсекаем шум)
MIN_AREA = 10

# Аппроксимация контура (чем больше, тем грубее контур)
# 0.001 — очень точно, 0.01 — грубо, но меньше точек
APPROX_EPSILON = 0.001


def mask_to_yolo_polygons(mask_path: Path):
    """Читает PNG-маску, возвращает список полигонов в YOLO-формате."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    h, w = mask.shape

    # Бинаризация
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Находим внешние контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue  # отсекаем мелкий шум

        # Упрощаем контур (меньше точек — быстрее обучение)
        epsilon = APPROX_EPSILON * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Нужно минимум 3 точки для полигона
        if len(approx) < 3:
            continue

        # Нормализация координат
        points = approx.reshape(-1, 2).astype(float)
        points[:, 0] /= w
        points[:, 1] /= h
        points = np.clip(points, 0.0, 1.0)

        # Формируем строку: class_id + координаты
        coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in points])
        lines.append(f"0 {coords}")

    return lines


def main():
    YOLO_DIR.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(LABELS_DIR.rglob("*.png"))
    print(f"Найдено масок: {len(mask_files)}")

    total_objects = 0
    empty_files = 0

    for mask_path in tqdm(mask_files):
        # Сохраняем ту же структуру папок, что и у масок
        rel_path = mask_path.relative_to(LABELS_DIR)
        out_path = YOLO_DIR / rel_path.with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines = mask_to_yolo_polygons(mask_path)

        if lines:
            out_path.write_text("\n".join(lines), encoding="utf-8")
            total_objects += len(lines)
        else:
            # Пустой файл — негативный пример (срез без очага)
            out_path.write_text("", encoding="utf-8")
            empty_files += 1

    print(f"\nГотово!")
    print(f"   Всего файлов аннотаций: {len(mask_files)}")
    print(f"   С очагами:              {len(mask_files) - empty_files}")
    print(f"   Пустых (негативных):    {empty_files}")
    print(f"   Всего полигонов:        {total_objects}")
    print(f"   Папка с аннотациями:    {YOLO_DIR}")


if __name__ == "__main__":
    main()