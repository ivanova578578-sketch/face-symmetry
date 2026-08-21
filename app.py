import streamlit as st
import cv2
import numpy as np

# Настройка внешнего вида страницы
st.set_page_config(page_title="Симметрия лица", page_icon="👤", layout="centered")

st.title("👤 Двойники в твоем лице")
st.write("Загрузи фото, чтобы увидеть, как бы ты выглядел, если бы твоё лицо было абсолютно симметричным!")

# Загрузка файла пользователем
uploaded_file = st.file_uploader("Выбери фотографию (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Читаем файл в формат OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    h, w, _ = img.shape
    
    # Геометрический центр кадра
    true_mid = w // 2
    
    # Ползунок двигает само фото (смещение влево/вправо относительно центра)
    max_shift = w // 3
    shift = st.slider("📐 Подвинь лицо влево или вправо, чтобы поймать центр носа:", 
                      min_value=-max_shift, max_value=max_shift, value=0, step=1)
    
    # Сдвигаем картинку по горизонтали
    M = np.float32([[1, 0, -shift],])
    img_shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Умная цветокоррекция яркости (CLAHE)
    lab = cv2.cvtColor(img_shifted, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    img_corrected = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Режем ровно пополам
    left_half = img_corrected[:, :true_mid]
    left_half_flipped = cv2.flip(left_half, 1)
    left_face = np.hstack((left_half, left_half_flipped))
    
    right_half = img_corrected[:, true_mid:]
    right_half_flipped = cv2.flip(right_half, 1)
    right_face = np.hstack((right_half_flipped, right_half))
    
    # --- ЖЕСТКОЕ ВЫРАВНИВАНИЕ РАЗМЕРА И ВЫСОТЫ ---
    # Принудительно делаем картинки квадратными (например, 600x600 пикселей)
    # Это полностью уберет разницу в высоте из-за текста или фона
    side_size = 600
    left_face_final = cv2.resize(left_face, (side_size, side_size))
    right_face_final = cv2.resize(right_face, (side_size, side_size))
    # ---------------------------------------------
    
    # Конвертируем в RGB для сайта
    left_face_rgb = cv2.cvtColor(left_face_final, cv2.COLOR_BGR2RGB)
    right_face_rgb = cv2.cvtColor(right_face_final, cv2.COLOR_BGR2RGB)
    
    # Отображение результатов на сайте
    st.write("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Левая сторона 🕶️")
        st.image(left_face_rgb, use_container_width=True)
        _, img_encoded = cv2.imencode('.jpg', left_face_final)
        st.download_button(label="📥 Скачать левое лицо", data=img_encoded.tobytes(), file_name="left_symmetry.jpg", mime="image/jpeg")
        
    with col2:
        st.subheader("Правая сторона ✨")
        st.image(right_face_rgb, use_container_width=True)
        _, img_encoded = cv2.imencode('.jpg', right_face_final)
        st.download_button(label="📥 Скачать правое лицо", data=img_encoded.tobytes(), file_name="right_symmetry.jpg", mime="image/jpeg")
