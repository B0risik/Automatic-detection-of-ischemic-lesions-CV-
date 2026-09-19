from ultralytics import YOLO
from pathlib import Path

model = YOLO(r"best_isles_250.pt")

# Предсказание на всех валидационных изображениях
val_dir = Path(r"C:\Ischemic Stroke Lesion Segmentation\yolo_dataset\images\val")
results = model.predict(
    source=str(val_dir),
    save=True,           # сохранить с нарисованными масками
    conf=0.15,           # низкий порог — на маленьком датасете модель неуверенная
    iou=0.5,
    name="predict_val",
)

print(f"Обработано изображений: {len(results)}")
for r in results[:3]:
    n = len(r.boxes) if r.boxes is not None else 0
    print(f"  {Path(r.path).name}: найдено очагов — {n}")