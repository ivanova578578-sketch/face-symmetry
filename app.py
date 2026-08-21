import streamlit as st
import cv2
import numpy as np
import os

# Настройка внешнего вида страницы
st.set_page_config(page_title="Симметрия лица", page_icon="👤", layout="centered")

st.title("👤 Двойники в твоем лице")
st.write("Загрузи фото, и умный алгоритм автоматически найдет центр твоего лица!")

# Загружаем классический встроенный детектор лиц OpenCV
@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_cascade()

# Загрузка файла пользователем
uploaded_file = st.file_uploader("Выбери фотографию (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Читаем файл в формат OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    # Центр лица по умолчанию (геометрическая середина кадра)
    auto_mid_x = w // 2
    
    # Пробуем найти лицо встроенными средствами
    if face_cascade is not None and not face_cascade.empty():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        if len(faces) > 0:
            # Берем первое найденное лицо (координаты: x, y, ширина, высота)
            fx, fy, fw, fh = faces[0]
            # Вычисляем точный геометрический центр этого лица по горизонтали
            auto_mid_x = int(fx + (fw / 2))
            st.success("🎯 Центр лица успешно найден автоматически! Выровняли разрез по переносице.")
        else:
            st.info("💡 Лицо не распознано автоматически (возможно, из-за наклона). Подправь ползунок вручную:")
    
    # Ползунок автоматически прыгает в найденную координату центра лица!
    mid_x = st.slider("📐 Смести линию разреза, если хочешь подкорректировать результат:", 
                      min_value=1, max_value=w-1, value=auto_mid_x, step=1)
    
    # Умная цветокоррекция яркости (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    img_corrected = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Обработка левой половины
    left_half = img_corrected[:, :mid_x]
    left_half_flipped = cv2.flip(left_half, 1)
    left_face = np.hstack((left_half, left_half_flipped))
    
    # Обработка правой половины
    right_half = img_corrected[:, mid_x:]
    right_half_flipped = cv2.flip(right_half, 1)
    right_face = np.hstack((right_half_flipped, right_half))
    
    # Выравнивание размеров для экрана сайта
    target_w = 400
    target_h = int(h * (target_w / w))
    
    left_face_disp = cv2.resize(left_face, (target_w, target_h))
    right_face_disp = cv2.resize(right_face, (target_w, target_h))
    
    left_face_rgb = cv2.cvtColor(left_face_disp, cv2.COLOR_BGR2RGB)
    right_face_rgb = cv2.cvtColor(right_face_disp, cv2.COLOR_BGR2RGB)
    
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
