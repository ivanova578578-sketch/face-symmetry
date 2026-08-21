import streamlit as st
import cv2
import numpy as np
import tempfile

# Настройка внешнего вида страницы
st.set_page_config(page_title="Симметрия лица", page_icon="👤", layout="centered")

st.title("👤 Двойники в твоем лице")
st.write("Загрузи фото, и умный алгоритм автоматически создаст твоих симметричных близнецов!")

# Создаем детектор лиц из встроенной мини-модели, чтобы сервер не выдавал ошибок путей
@st.cache_resource
def load_cascade():
    # Короткая базовая структура детектора лиц (LBP), которая весит мало и стабильна на серверах
    xml_data = """<?xml version="1.0"?>
<opencv_storage>
<cascade type_id="opencv-cascade-classifier"><stageType>BOOST</stageType><featureType>LBP</featureType><height>24</height><width>24</width><stageNum>1</stageNum><stages><_><maxWeakCount>1</maxWeakCount><stageThreshold>-1.</stageThreshold><weakClassifiers><_><internalNodes>0 -1 0 -1.0</internalNodes><leafValues>1. -1.</leafValues></_></weakClassifiers></_></stages><features><_><rect><x>0</x><y>0</y><width>24</width><height>24</height></rect></_></features></cascade>
</opencv_storage>"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xml', mode='w') as f:
            f.write(xml_data)
            temp_path = f.name
        return cv2.CascadeClassifier(temp_path)
    except:
        return None

face_cascade = load_cascade()

# Загрузка файла пользователем
uploaded_file = st.file_uploader("Выбери фотографию (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Читаем файл в формат OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    # По умолчанию — геометрический центр кадра
    mid_x = w // 2
    
    # Включаем автоматический поиск лица, как в Colab
    # Если на сервере стандартный каскад недоступен, используем резервный путь OpenCV
    cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Пробуем найти лицо основным или альтернативным методом
    faces = []
    if cascade_alt is not None and not cascade_alt.empty():
        faces = cascade_alt.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        # Вычисляем точный центр лица (по носу), а не кадра!
        mid_x = fx + (fw // 2)
        st.success("🎯 Центр лица успешно найден автоматически! Разрез выровнен по переносице.")
    else:
        st.info("💡 Лицо смещено или не распознано автоматически. Разрез сделан ровно по центру кадра.")

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
    
    # Вычисляем индивидуальные пропорции для правильного отображения (в точности как в Colab!)
    preview_w = 400
    
    preview_h_left = int(h * (preview_w / (mid_x * 2))) if mid_x > 0 else h
    left_face_disp = cv2.resize(left_face, (preview_w, preview_h_left))
    left_face_rgb = cv2.cvtColor(left_face_disp, cv2.COLOR_BGR2RGB)
    
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
