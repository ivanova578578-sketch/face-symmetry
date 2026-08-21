import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Настройка внешнего вида страницы
st.set_page_config(page_title="Симметрия лица", page_icon="👤", layout="centered")

st.title("👤 Двойники в твоем лице")
st.write("Загрузи фото, и умная нейросеть автоматически найдет центр твоего лица!")

# Инициализируем нейросеть Google MediaPipe для поиска точек лица
@st.cache_resource
def load_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

face_mesh = load_mesh()

# Загрузка файла пользователем
uploaded_file = st.file_uploader("Выбери фотографию (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Читаем файл в формат OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    # Переводим в RGB для нейросети
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    # Определяем центр лица по умолчанию (середина кадра)
    auto_mid_x = w // 2
    
    if results.multi_face_landmarks:
        # Нейросеть нашла лицо! Берем ключевые осевые точки:
        # Индексы точек MediaPipe: 168 (переносица), 1 (кончик носа), 152 (подбородок)
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Получаем координаты X для этих точек (они в процентах от 0 до 1, умножаем на ширину w)
        bridge_x = int(landmarks[168].x * w)
        nose_x = int(landmarks[1].x * w)
        chin_x = int(landmarks[152].x * w)
        
        # Находим среднее арифметическое центральной оси лица
        auto_mid_x = int((bridge_x + nose_x + chin_x) / 3)
        st.success("🎯 Нейросеть успешно нашла точный центр лица по переносице, носу и подбородку!")
    else:
        st.warning("⚠️ Лицо не распознано автоматически. Используется центр кадра. Ты можешь настроить его вручную:")

    # Ползунок подстраивается автоматически под точку носа, найденную нейросетью!
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
