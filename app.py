import streamlit as st
import cv2
import numpy as np

# Настройка внешнего вида страницы
st.set_page_config(page_title="Симметрия лица", page_icon="👤", layout="centered")

st.title("👤 Двойники в твоем лице")
st.write("Загрузи фото, и умный алгоритм автоматически создаст твоих симметричных близнецов!")

# Загрузка файла пользователем
uploaded_file = st.file_uploader("Выбери фотографию (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Читаем файл в формат OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    # --- УМНЫЙ МАТЕМАТИЧЕСКИЙ АВТОПОИСК ЦЕНТРА ЛИЦА ---
    # Переводим в Ч/Б и размываем, чтобы убрать мелкий шум фона
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Считаем среднюю яркость по вертикальным колонкам в центральной зоне кадра
    # Обычно нос и переносица — самые освещенные вертикальные оси на портрете
    zone_start = int(w * 0.35)
    zone_end = int(w * 0.65)
    column_brightness = np.mean(blurred[:, zone_start:zone_end], axis=0)
    
    # Находим индекс колонки с максимальной яркостью (центр носа)
    if len(column_brightness) > 0:
        mid_x = zone_start + np.argmax(column_brightness)
        st.success("🎯 Центр лица успешно определен алгоритмом по оси носа!")
    else:
        mid_x = w // 2
        st.info("💡 Разрез выполнен ровно по центру кадра.")
    # --------------------------------------------------

    # Умная цветокоррекция яркости (CLAHE) — убирает боковые тени, как в Colab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    img_corrected = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # --- ТОЧНАЯ МАТЕМАТИКА ИЗ ВАШЕГО COLAB-КОДА ---
    # Обработка левой половины (размер итогового фото будет строго mid_x * 2)
    left_half = img_corrected[:, :mid_x]
    left_half_flipped = cv2.flip(left_half, 1)
    left_face = np.hstack((left_half, left_half_flipped))
    
    # Обработка правой половины (размер итогового фото будет строго (w - mid_x) * 2)
    right_half = img_corrected[:, mid_x:]
    right_half_flipped = cv2.flip(right_half, 1)
    right_face = np.hstack((right_half_flipped, right_half))
    
    # Вычисляем индивидуальные пропорции для правильного отображения (как в Colab!)
    preview_w = 400
    
    # Высота для левого лица
    preview_h_left = int(h * (preview_w / (mid_x * 2))) if mid_x > 0 else h
    left_face_disp = cv2.resize(left_face, (preview_w, preview_h_left))
    left_face_rgb = cv2.cvtColor(left_face_disp, cv2.COLOR_BGR2RGB)
    
    # Высота для правого лица
    preview_h_right = int(h * (preview_w / ((w - mid_x) * 2))) if (w - mid_x) > 0 else h
    right_face_disp = cv2.resize(right_face, (preview_w, preview_h_right))
    right_face_rgb = cv2.cvtColor(right_face_disp, cv2.COLOR_BGR2RGB)
    # -------------------------------------------------------------------------
    
    # Отображение результатов на сайте
    st.write("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Левая сторона 🕶️")
        st.image(left_face_rgb, use_container_width=True)
        _, img_encoded = cv2.imencode('.jpg', left_face)
        st.download_button(label="📥 Скачать левое лицо", data=img_encoded.tobytes(), file_name="left_symmetry.jpg", mime="image/jpeg")
        
    with col2:
        st.subheader("Правая сторона ✨")
        st.image(right_face_rgb, use_container_width=True)
        _, img_encoded = cv2.imencode('.jpg', right_face)
        st.download_button(label="📥 Скачать правое лицо", data=img_encoded.tobytes(), file_name="right_symmetry.jpg", mime="image/jpeg")
