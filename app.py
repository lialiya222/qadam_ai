import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(
    page_title="QADAM AI — ЛФК Ассистент",
    page_icon="🦾",
    layout="wide"
)

st.title("🦾 QADAM AI — Система реабилитации плечевого сустава")
st.caption("AI-ассистент для контроля техники выполнения ЛФК (Отведение плеча)")

# --- АБСОЛЮТНЫЕ ПУТИ ДЛЯ НАДЕЖНОЙ ЗАГРУЗКИ МОДЕЛИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    with st.spinner("Загрузка файла модели AI (выполняется один раз)..."):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            st.error(f"Ошибка загрузки файла модели: {e}")

# --- ИНИЦИАЛИЗАЦИЯ ДЕТЕКТОРА ПОЗЫ ---
@st.cache_resource
def load_detector():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Файл модели не найден по пути: {MODEL_PATH}")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)

try:
    detector = load_detector()
except Exception as e:
    st.error(f"Ошибка инициализации детектора позы: {e}")
    detector = None

# --- ФУНКЦИЯ РАСЧЕТА УГЛА МЕЖДУ 3 ТОЧКАМИ ---
def calculate_angle(a, b, c):
    a = np.array(a)  # Таз
    b = np.array(b)  # Плечо (вершина)
    c = np.array(c)  # Локоть

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

# --- БОКОВАЯ ПАНЕЛЬ С НАСТРОЙКАМИ ---
st.sidebar.header("⚙️ Настройки и Параметры")
arm_choice = st.sidebar.selectbox("Выберите рабочую руку:", ["Левая рука", "Правая рука"])
run_app = st.sidebar.checkbox("Включить веб-камеру", value=False)

ANGLE_DOWN = st.sidebar.slider("Угол руки опущенной (< degree):", 10, 50, 30)
ANGLE_UP = st.sidebar.slider("Целевой угол подъема (> degree):", 50, 110, 70)

col1, col2 = st.columns([3, 1])

with col1:
    frame_window = st.empty()

with col2:
    st.subheader("📊 Показатели")
    rep_metric = st.empty()
    angle_metric = st.empty()
    status_metric = st.empty()

counter = 0
stage = "DOWN"

# --- ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ВИДЕОПОТОКА ---
if run_app and detector is not None:
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and run_app:
        ret, frame = cap.read()
        if not ret:
            st.error("Не удалось получить видеопоток с веб-камеры.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect(mp_image)
        current_angle = 0

        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            landmarks = detection_result.pose_landmarks[0]

            if arm_choice == "Левая рука":
                hip_idx, shoulder_idx, elbow_idx = 23, 11, 13
            else:
                hip_idx, shoulder_idx, elbow_idx = 24, 12, 14

            hip = [landmarks[hip_idx].x * w, landmarks[hip_idx].y * h]
            shoulder = [landmarks[shoulder_idx].x * w, landmarks[shoulder_idx].y * h]
            elbow = [landmarks[elbow_idx].x * w, landmarks[elbow_idx].y * h]

            current_angle = calculate_angle(hip, shoulder, elbow)

            if current_angle > ANGLE_UP:
                stage = "UP"
                skeleton_color = (0, 255, 0)
            elif current_angle < ANGLE_DOWN and stage == "UP":
                stage = "DOWN"
                counter += 1
                skeleton_color = (0, 0, 255)
            else:
                skeleton_color = (255, 255, 255)

            cv2.line(frame, (int(hip[0]), int(hip[1])), (int(shoulder[0]), int(shoulder[1])), skeleton_color, 4)
            cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), skeleton_color, 4)

            for pt in [hip, shoulder, elbow]:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, (255, 0, 0), -1)

            cv2.putText(
                frame, f"{int(current_angle)} deg",
                (int(shoulder[0]) + 15, int(shoulder[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )

        rep_metric.metric("Счетчик повторов", f"{counter}")
        angle_metric.metric("Текущий угол", f"{int(current_angle)}°")
        status_metric.metric("Состояние", stage)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()
else:
    st.info("Поставьте галочку «Включить веб-камеру» на боковой панели, чтобы начать отслеживание.")