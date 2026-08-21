import streamlit as st
import cv2
import numpy as np

# Настройка внешнего вида страницы
st.set_page_config(page_title="Симметрия лица", page_icon="👤", layout="centered")

st.title("👤 Двойники в твоем лице")
st.write("Загрузи фото анфас, чтобы увидеть, как бы ты выглядел, если бы твоё лицо было абсолютно симметричным!")

# Загрузка файла пользователем
uploaded_file = st.file_uploader("Выбери фотографию (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Читаем файл в формат OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    h, w, _ = img.shape
    
    # Геометрический центр картинки по умолчанию
    default_mid = w // 2
    
    # Плавный интерактивный ползунок для ручной настройки центра носа
    mid_x = st.slider("📐 Наведи ползунок точно на центр своего носа:", 
                      min_value=1, max_value=w-1, value=default_mid, step=1)
    
    # Умная цветокоррекция яркости (CLAHE) — убирает боковые тени
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    img_corrected = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Обработка левой половины (размер итогового фото будет строго mid_x * 2)
    left_half = img_corrected[:, :mid_x]
    left_half_flipped = cv2.flip(left_half, 1)
    left_face = np.hstack((left_half, left_half_flipped))
    
    # Обработка правой половины (размер итогового фото будет строго (w - mid_x) * 2)
    right_half = img_corrected[:, mid_x:]
    right_half_flipped = cv2.flip(right_half, 1)
    right_face = np.hstack((right_half_flipped, right_half))
    
    # --- МАТЕМАТИЧЕСКОЕ ВЫРАВНИВАНИЕ ЭКРАНА ---
    # Приводим оба лица на экране к одной красивой ширине (например, 400 пикселей)
    target_w = 400
    target_h = int(h * (target_w / w)) # Высота рассчитывается автоматически по пропорциям оригинала
    
    left_face_disp = cv2.resize(left_face, (target_w, target_h))
    right_face_disp = cv2.resize(right_face, (target_w, target_h))
    
    # Переводим в RGB для правильного отображения цветов на сайте
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
