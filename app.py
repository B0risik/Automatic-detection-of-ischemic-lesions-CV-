"""
Streamlit-приложение для инференса модели выявления ишемических очагов.
Загружает NIfTI (DWI + ADC) или 2-канальный PNG, прогоняет через YOLOv8-seg,
показывает маску очага и формирует отчёт.
"""
import io
import os
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
import nibabel as nib
import cv2
import tempfile
import time
import pandas as pd

from ultralytics import YOLO

# ==================== КОНФИГ ====================
st.set_page_config(
    page_title="Выявление ишемических очагов",
    page_icon="🧠",
    layout="wide",
)

MODEL_PATH = Path(r"C:\Ischemic Stroke Lesion Segmentation\best_isles_250.pt")
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.5
TARGET_SIZE = 256


# ==================== ФУНКЦИИ ====================
@st.cache_resource
def load_model(path: str):
    if not Path(path).exists():
        return None
    return YOLO(path)


def normalize_mri(volume: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    if p99 - p1 < 1e-6:
        return np.zeros_like(volume, dtype=np.uint8)
    volume = (volume - p1) / (p99 - p1) * 255.0
    return volume.astype(np.uint8)


def make_rgb(dwi_slice: np.ndarray, adc_slice: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Создаёт 2-канальное RGB: R=DWI, G=ADC."""
    dwi_norm = np.rot90(normalize_mri(dwi_slice))
    adc_norm = np.rot90(normalize_mri(adc_slice))

    dwi_img = Image.fromarray(dwi_norm).resize((size, size), Image.BILINEAR)
    adc_img = Image.fromarray(adc_norm).resize((size, size), Image.BILINEAR)

    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = np.array(dwi_img)
    rgb[..., 1] = np.array(adc_img)
    return rgb


def to_bgr(rgb: np.ndarray) -> np.ndarray:
    """RGB → BGR. Ultralytics ожидает BGR при подаче numpy-массива."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_nifti(uploaded_file):
    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        img = nib.load(tmp_path)
        data = img.get_fdata()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return data


def check_dwi_adc_consistency(dwi_slice, adc_slice, mask):
    if mask.sum() == 0:
        return None

    dwi_inside  = dwi_slice[mask > 0].mean()
    adc_inside  = adc_slice[mask > 0].mean()
    dwi_outside = dwi_slice[mask == 0].mean()
    adc_outside = adc_slice[mask == 0].mean()

    return {
        "dwi_inside":  float(dwi_inside),
        "dwi_outside": float(dwi_outside),
        "adc_inside":  float(adc_inside),
        "adc_outside": float(adc_outside),
        "dwi_ratio":   float(dwi_inside / (dwi_outside + 1e-6)),
        "adc_ratio":   float(adc_inside / (adc_outside + 1e-6)),
    }


# ==================== UI ====================
st.title("🧠 Автоматическое выявление ишемических очагов на МРТ")
st.markdown(
    """
    Система сопоставляет **DWI** и **ADC** последовательности, находит острые 
    ишемические очаги и выделяет их границы.

    **Ключевой принцип:** очаг — это область, где DWI **яркий**, а ADC **тёмный**.
    """
)

with st.sidebar:
    st.header("⚙️ Настройки")

    conf_threshold = st.slider(
        "Порог уверенности модели",
        min_value=0.05, max_value=0.90,
        value=DEFAULT_CONF, step=0.05,
        help="Чем ниже — тем больше очагов найдёт модель, но больше ложных срабатываний"
    )

    iou_threshold = st.slider(
        "Порог IoU (NMS)",
        min_value=0.1, max_value=0.9,
        value=DEFAULT_IOU, step=0.05,
    )

    show_verification = st.checkbox(
        "Проверять логику DWI/ADC", value=True,
    )

    st.markdown("---")
    st.markdown("**Модель:** YOLOv8n-seg")
    st.markdown("**Датасет:** ISLES 2022 (250 пациентов)")
    st.markdown("**Mask mAP50:** 0.561")

model = load_model(str(MODEL_PATH))
if model is None:
    st.error(f"❌ Модель не найдена: `{MODEL_PATH}`")
    st.stop()

st.success(f"✅ Модель загружена: `{MODEL_PATH.name}`")

st.header("📁 Загрузка исследования")

input_mode = st.radio(
    "Формат входных данных:",
    ["NIfTI (DWI + ADC)", "Готовое PNG (2-канальное)"],
    horizontal=True,
)

dwi_data = None
adc_data = None
png_file = None

if input_mode == "NIfTI (DWI + ADC)":
    col1, col2 = st.columns(2)
    with col1:
        dwi_file = st.file_uploader("DWI (.nii или .nii.gz)", type=["nii", "gz"], key="dwi")
    with col2:
        adc_file = st.file_uploader("ADC (.nii или .nii.gz)", type=["nii", "gz"], key="adc")

    if dwi_file and adc_file:
        dwi_data = load_nifti(dwi_file)
        adc_data = load_nifti(adc_file)
        st.success(f"Загружено: DWI {dwi_data.shape}, ADC {adc_data.shape}")
else:
    png_file = st.file_uploader("2-канальное PNG (R=DWI, G=ADC)", type=["png", "jpg", "jpeg"])
    if png_file:
        rgb_array = np.array(Image.open(png_file).convert("RGB"))
        st.success(f"Загружено: {rgb_array.shape}")


# ==================== АНАЛИЗ NIfTI ====================
if input_mode == "NIfTI (DWI + ADC)" and dwi_data is not None:
    n_slices = dwi_data.shape[2]
    st.header("🔬 Выбор среза")

    slice_idx = st.slider(
        "Срез (по оси Z)",
        min_value=0, max_value=n_slices - 1,
        value=n_slices // 2,
    )

    dwi_slice = dwi_data[:, :, slice_idx]
    adc_slice = adc_data[:, :, slice_idx]

    rgb = make_rgb(dwi_slice, adc_slice)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**DWI** (яркий = очаг)")
        st.image(np.rot90(normalize_mri(dwi_slice)), use_container_width=True, clamp=True)
    with col2:
        st.markdown("**ADC** (тёмный = очаг)")
        st.image(np.rot90(normalize_mri(adc_slice)), use_container_width=True, clamp=True)
    with col3:
        st.markdown("**Совмещённое** (R=DWI, G=ADC)")
        st.image(rgb, use_container_width=True)

    # --- Инференс одного среза ---
    if st.button("🔍 Найти очаги на выбранном срезе", type="primary", use_container_width=True):
        with st.spinner("Анализирую..."):
            result = model.predict(
                to_bgr(rgb),                  # ← RGB → BGR
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )[0]

        st.header("📊 Результат анализа")

        if result.masks is None or len(result.masks) == 0:
            st.warning("⚠️ Очагов не найдено.")
            st.info("Попробуйте снизить порог уверенности в настройках слева.")
        else:
            n_lesions = len(result.masks)
            st.success(f"✅ Найдено очагов: **{n_lesions}**")

            annotated = rgb.copy()
            h, w = rgb.shape[:2]
            total_area = 0

            for i, mask in enumerate(result.masks.data.cpu().numpy()):
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 0.5
                total_area += mask_bool.sum()

                annotated[mask_bool] = (
                    annotated[mask_bool] * 0.4 +
                    np.array([255, 0, 0]) * 0.6
                ).astype(np.uint8)

                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated, contours, -1, (255, 255, 0), 2)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Исходное (R=DWI, G=ADC)**")
                st.image(rgb, use_container_width=True)
            with col2:
                st.markdown("**Найденные очаги** (красный)")
                st.image(annotated, use_container_width=True)

            st.subheader("Детали по очагам")
            confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

            lesion_data = []
            for i in range(n_lesions):
                conf = float(confs[i]) if i < len(confs) else 0.0
                mask_i = result.masks.data[i].cpu().numpy()
                mask_i = cv2.resize(mask_i, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                area_px = int(mask_i.sum())
                area_pct = 100 * area_px / (h * w)

                lesion_data.append({
                    "Очаг": f"#{i + 1}",
                    "Уверенность": f"{conf:.2%}",
                    "Площадь (px)": area_px,
                    "Площадь (%)": f"{area_pct:.2f}%",
                })

            st.dataframe(lesion_data, use_container_width=True)

            if show_verification:
                st.subheader("🧪 Проверка логики DWI/ADC")

                combined_mask = np.zeros((h, w), dtype=np.uint8)
                for i in range(n_lesions):
                    m = result.masks.data[i].cpu().numpy()
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                    combined_mask[m] = 1

                h_orig, w_orig = dwi_slice.shape
                combined_mask_orig = cv2.resize(
                    combined_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
                )

                dwi_orig = np.rot90(dwi_slice)
                adc_orig = np.rot90(adc_slice)

                stats = check_dwi_adc_consistency(dwi_orig, adc_orig, combined_mask_orig)

                if stats:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "DWI внутри / снаружи",
                            f"{stats['dwi_ratio']:.2f}x",
                            delta="должно быть > 1.0",
                            delta_color="normal" if stats['dwi_ratio'] > 1.0 else "inverse",
                        )
                    with col2:
                        st.metric(
                            "ADC внутри / снаружи",
                            f"{stats['adc_ratio']:.2f}x",
                            delta="должно быть < 1.0",
                            delta_color="normal" if stats['adc_ratio'] < 1.0 else "inverse",
                        )

                    if stats['dwi_ratio'] > 1.0 and stats['adc_ratio'] < 1.0:
                        st.success(
                            "✅ Логика подтверждена: DWI ярче, ADC темнее. "
                            "Соответствует острому ишемическому очагу."
                        )
                    else:
                        st.warning(
                            "⚠️ Логика не подтверждена. Рекомендуется проверка специалистом."
                        )

            st.subheader("📋 Отчёт")
            st.markdown(f"""
**Срез:** #{slice_idx}  
**Модель:** YOLOv8n-seg (ISLES 2022, 250 пациентов)  
**Порог уверенности:** {conf_threshold:.2f}  
**IoU:** {iou_threshold:.2f}

**Результат:**
- Найдено очагов: **{n_lesions}**
- Общая площадь: **{int(total_area)} px** ({100 * total_area / (h * w):.2f}% среза)
- Средняя уверенность: **{np.mean(confs):.2%}**

**Рекомендация:** результаты носят вспомогательный характер.
            """)

            annotated_pil = Image.fromarray(annotated)
            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")
            st.download_button(
                "💾 Скачать результат (PNG)",
                data=buf.getvalue(),
                file_name=f"lesion_slice{slice_idx}.png",
                mime="image/png",
            )

    # ==================== АВТОПРОКРУТКА ====================
    st.markdown("---")
    st.header("🎬 Автоматический анализ всего исследования")

    col1, col2, col3 = st.columns(3)
    with col1:
        auto_conf = st.slider(
            "Порог для автопрокрутки",
            min_value=0.05, max_value=0.90,
            value=DEFAULT_CONF, step=0.05,
            key="auto_conf",
        )
    with col2:
        auto_iou = st.slider(
            "IoU для автопрокрутки",
            min_value=0.1, max_value=0.9,
            value=DEFAULT_IOU, step=0.05,
            key="auto_iou",
        )
    with col3:
        min_area = st.number_input(
            "Мин. площадь очага (px)",
            min_value=1, max_value=500, value=10, step=5,
            key="auto_min_area",
        )

    show_animation = st.checkbox(
        "Показывать анимацию", value=False, key="auto_anim",
    )

    if st.button("▶ Запустить автопрокрутку", type="primary", use_container_width=True):
        progress = st.progress(0.0, text="Анализирую...")
        status = st.empty()
        animation_slot = st.empty()

        results_auto = []
        lesion_slices = []
        total_lesions = 0
        total_area_px = 0
        all_detections = []

        start_time = time.time()

        for i in range(n_slices):
            progress.progress((i + 1) / n_slices, text=f"Срез {i + 1} / {n_slices}")

            dwi_s = dwi_data[:, :, i]
            adc_s = adc_data[:, :, i]

            if dwi_s.max() < 1e-3 and adc_s.max() < 1e-3:
                continue

            rgb_i = make_rgb(dwi_s, adc_s)
            res = model.predict(
                to_bgr(rgb_i),                 # ← RGB → BGR
                conf=auto_conf,
                iou=auto_iou,
                verbose=False,
            )[0]

            n_les = 0
            slice_area = 0
            if res.masks is not None and len(res.masks) > 0:
                confs_i = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
                for j, mask in enumerate(res.masks.data.cpu().numpy()):
                    m = cv2.resize(
                        mask, (TARGET_SIZE, TARGET_SIZE),
                        interpolation=cv2.INTER_NEAREST
                    ) > 0.5
                    area = int(m.sum())
                    if area < min_area:
                        continue
                    n_les += 1
                    slice_area += area
                    all_detections.append({
                        "slice": i,
                        "lesion_idx": n_les,
                        "confidence": float(confs_i[j]) if j < len(confs_i) else 0.0,
                        "area_px": area,
                    })

            results_auto.append({"slice": i, "n_lesions": n_les})

            if n_les > 0:
                lesion_slices.append(i)
                total_lesions += n_les
                total_area_px += slice_area
                status.info(f"🔴 Срез {i}: найдено очагов — {n_les}, площадь — {slice_area} px")

            if show_animation:
                annotated_i = rgb_i.copy()
                if res.masks is not None and len(res.masks) > 0:
                    for mask in res.masks.data.cpu().numpy():
                        m = cv2.resize(
                            mask, (TARGET_SIZE, TARGET_SIZE),
                            interpolation=cv2.INTER_NEAREST
                        ) > 0.5
                        annotated_i[m] = (
                            annotated_i[m] * 0.4 +
                            np.array([255, 0, 0]) * 0.6
                        ).astype(np.uint8)
                animation_slot.image(
                    annotated_i,
                    caption=f"Срез {i} — очагов: {n_les}",
                    use_container_width=True,
                )
                time.sleep(0.15)

        elapsed = time.time() - start_time
        progress.empty()
        status.empty()
        animation_slot.empty()

        st.success(
            f"✅ Обработано **{n_slices}** срезов за **{elapsed:.1f} сек**. "
            f"Найдено очагов: **{total_lesions}** на **{len(lesion_slices)}** срезах."
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Всего очагов", total_lesions)
        with col2:
            st.metric("Срезов с очагами", f"{len(lesion_slices)} / {n_slices}")
        with col3:
            st.metric("Общая площадь", f"{total_area_px} px")
        with col4:
            st.metric("Время анализа", f"{elapsed:.1f} сек")

        if lesion_slices:
            st.markdown("**Срезы с очагами:**")
            st.code(", ".join(map(str, lesion_slices)), language=None)

            st.markdown("**Распределение очагов по срезам:**")
            df_hist = pd.DataFrame(results_auto)
            df_hist = df_hist[df_hist["n_lesions"] > 0]
            if not df_hist.empty:
                st.bar_chart(df_hist.set_index("slice")["n_lesions"])

            st.markdown("**Все найденные очаги:**")
            df_det = pd.DataFrame(all_detections)
            df_det["confidence"] = df_det["confidence"].apply(lambda x: f"{x:.2%}")
            st.dataframe(df_det, use_container_width=True)

            csv = df_det.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Скачать отчёт (CSV)",
                data=csv,
                file_name="lesions_report.csv",
                mime="text/csv",
            )
        else:
            st.info("ℹ️ Очаги не найдены ни на одном срезе.")


# ==================== АНАЛИЗ PNG ====================
elif input_mode == "Готовое PNG (2-канальное)" and png_file:
    rgb = np.array(Image.open(png_file).convert("RGB"))
    st.image(rgb, caption="Загруженное изображение (R=DWI, G=ADC)", use_container_width=True)

    if st.button("🔍 Найти очаги", type="primary"):
        with st.spinner("Анализирую..."):
            result = model.predict(
                to_bgr(rgb),                   # ← RGB → BGR
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )[0]

        if result.masks is None or len(result.masks) == 0:
            st.warning("⚠️ Очагов не найдено.")
        else:
            st.success(f"✅ Найдено очагов: {len(result.masks)}")
            annotated = rgb.copy()
            for mask in result.masks.data.cpu().numpy():
                m = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
                annotated[m] = (annotated[m] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)

            col1, col2 = st.columns(2)
            with col1:
                st.image(rgb, caption="Исходное", use_container_width=True)
            with col2:
                st.image(annotated, caption="Найденные очаги", use_container_width=True)