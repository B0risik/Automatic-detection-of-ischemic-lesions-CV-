"""
Полный пайплайн подготовки датасета ISLES 2022 для YOLOv8.
Запускает последовательно:
  1. prepare_dataset.py — NIfTI в 2-канальные PNG + маски
  2. make_yolo_labels.py — PNG-маски в YOLO .txt аннотации
  3. split_dataset.py — разделение train/val по пациентам

"""
import subprocess
import sys
import time
import shutil
from pathlib import Path

# ==================== КОНФИГ ====================
PROJECT_DIR = Path(__file__).parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"

# Папка со скриптами пайплайна (относительно корня проекта)
SCRIPTS_DIR = PROJECT_DIR / "scripts"

# Папки, которые нужно очистить перед запуском
DIRS_TO_CLEAN = [
    PROJECT_DIR / "yolo_dataset_raw",
    PROJECT_DIR / "yolo_dataset",
]

# Скрипты для последовательного запуска (лежат в SCRIPTS_DIR)
SCRIPTS = [
    ("prepare_dataset.py",   "NIfTI → PNG (2-канальные: DWI + ADC)"),
    ("make_yolo_labels.py",  "PNG-маски → YOLO .txt аннотации"),
    ("split_dataset.py",     "Разделение train/val по пациентам"),
]


# ==================== ФУНКЦИИ ====================
def print_header(text: str):
    """Печатает красивый заголовок."""
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def check_python():
    """Проверяет, что .venv существует и Python в нём работает."""
    if not VENV_PYTHON.exists():
        print(f"❌ Не найден Python в .venv: {VENV_PYTHON}")
        print("   Убедитесь, что виртуальное окружение создано.")
        sys.exit(1)

    result = subprocess.run(
        [str(VENV_PYTHON), "--version"],
        capture_output=True, text=True
    )
    print(f"✅ Python: {result.stdout.strip()}")
    print(f"   Путь: {VENV_PYTHON}")


def check_scripts():
    """Проверяет, что все скрипты пайплайна на месте."""
    print(f"\n📂 Папка со скриптами: {SCRIPTS_DIR}")
    if not SCRIPTS_DIR.exists():
        print(f"❌ Папка не найдена: {SCRIPTS_DIR}")
        print("   Создайте папку 'scripts' и положите туда три скрипта.")
        sys.exit(1)

    all_ok = True
    for script_name, _ in SCRIPTS:
        p = SCRIPTS_DIR / script_name
        if p.exists():
            print(f"   ✅ {script_name}")
        else:
            print(f"   ❌ {script_name} — НЕ НАЙДЕН")
            all_ok = False

    if not all_ok:
        sys.exit(1)


def clean_old_results():
    """Удаляет старые папки с результатами."""
    print_header("ОЧИСТКА СТАРЫХ РЕЗУЛЬТАТОВ")

    for d in DIRS_TO_CLEAN:
        if d.exists():
            size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 ** 2)
            print(f"🗑️  Удаляю {d.name} ({size_mb:.1f} MB)...")
            shutil.rmtree(d)
        else:
            print(f"⏭️  {d.name} не существует — пропуск")

    print("\n✅ Очистка завершена.")


def run_script(script_name: str, description: str, step: int, total: int):
    """Запускает один скрипт и ждёт его завершения."""
    print_header(f"ШАГ {step}/{total}: {description}")
    print(f"   Запуск: {script_name}\n")

    script_path = SCRIPTS_DIR / script_name    # ← ИЗМЕНЕНО
    if not script_path.exists():
        print(f"❌ Скрипт не найден: {script_path}")
        sys.exit(1)

    start = time.time()

    process = subprocess.Popen(
        [str(VENV_PYTHON), str(script_path)],
        cwd=str(PROJECT_DIR),                   # рабочая папка — корень проекта
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    for line in process.stdout:
        print(f"   │ {line.rstrip()}")

    process.wait()
    elapsed = time.time() - start

    if process.returncode != 0:
        print(f"\n❌ Ошибка в {script_name} (код {process.returncode})")
        print(f"   Время: {elapsed:.1f} сек")
        sys.exit(1)

    print(f"\n✅ {script_name} завершён за {elapsed:.1f} сек ({elapsed / 60:.1f} мин)")


def check_results():
    """Проверяет, что все результаты на месте."""
    print_header("ПРОВЕРКА РЕЗУЛЬТАТОВ")

    checks = [
        ("yolo_dataset_raw/images", "PNG-изображения (2-канальные)"),
        ("yolo_dataset_raw/labels", "PNG-маски"),
        ("yolo_dataset_raw/labels_yolo", "YOLO .txt аннотации"),
        ("yolo_dataset/images/train", "Train изображения"),
        ("yolo_dataset/images/val", "Val изображения"),
        ("yolo_dataset/labels/train", "Train аннотации"),
        ("yolo_dataset/labels/val", "Val аннотации"),
        ("yolo_dataset/data.yaml", "data.yaml"),
    ]

    all_ok = True
    for rel_path, desc in checks:
        p = PROJECT_DIR / rel_path
        if not p.exists():
            print(f"❌ {desc}: НЕ НАЙДЕНО ({rel_path})")
            all_ok = False
            continue

        if p.is_dir():
            n = len(list(p.glob("*")))
            print(f"✅ {desc}: {n} файлов")
        else:
            print(f"✅ {desc}: OK")

    # Статистика по train/val
    print("\n📊 Статистика:")
    for split in ["train", "val"]:
        img_dir = PROJECT_DIR / "yolo_dataset" / "images" / split
        lbl_dir = PROJECT_DIR / "yolo_dataset" / "labels" / split
        if img_dir.exists():
            n_img = len(list(img_dir.glob("*.png")))
            n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0

            n_with_lesion = 0
            if lbl_dir.exists():
                for f in lbl_dir.glob("*.txt"):
                    if f.stat().st_size > 0:
                        n_with_lesion += 1

            print(f"   {split.upper()}:")
            print(f"      Изображений: {n_img}")
            print(f"      Аннотаций:   {n_lbl}")
            print(f"      С очагами:   {n_with_lesion} ({100 * n_with_lesion / max(n_lbl, 1):.1f}%)")

    for split in ["train", "val"]:
        p = PROJECT_DIR / "yolo_dataset" / f"{split}_patients.txt"
        if p.exists():
            patients = p.read_text().strip().split("\n")
            print(f"   Пациентов в {split}: {len(patients)}")

    if not all_ok:
        print("\n⚠️  Некоторые проверки не прошли. Проверьте логи выше.")
        sys.exit(1)

    print("\n✅ Все проверки пройдены!")


def main():
    print_header("ПАЙПЛАЙН ПОДГОТОВКИ ДАТАСЕТА ISLES 2022")
    print(f"   Проект: {PROJECT_DIR}")
    print(f"   Python: {VENV_PYTHON}")

    check_python()
    check_scripts()
    clean_old_results()

    total_start = time.time()
    for i, (script, desc) in enumerate(SCRIPTS, start=1):
        run_script(script, desc, i, len(SCRIPTS))

    total_elapsed = time.time() - total_start

    check_results()

    print_header("ГОТОВО")
    print(f"⏱️  Общее время: {total_elapsed:.1f} сек ({total_elapsed / 60:.1f} мин)")
    print(f"📁 Результат: {PROJECT_DIR / 'yolo_dataset'}")
    print(f"\nСледующий шаг: обучение YOLOv8 (локально или в Google Colab)")


if __name__ == "__main__":
    main()