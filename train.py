from ultralytics import YOLO
import torch

# Проверка устройства
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Использую устройство: {device}")

# Загружаем предобученную модель для сегментации
model = YOLO("yolov8n-seg.pt")  # nano — самая быстрая

# Обучение
results = model.train(
    data=r"C:\Ischemic Stroke Lesion Segmentation\yolo_dataset\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="isles_yolov8n",
    patience=15,           # ранняя остановка, если 15 эпох без улучшения
    device=device,
    seed=42,
    plots=True,            # сохранять графики
    verbose=True,
)