# config.py
from imports import *

# Настройки MediaPipe
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'min_tracking_confidence': 0.5,
    'min_detection_confidence': 0.5,
    'model_complexity': 1
}

# Настройки распознавания лиц
FACE_RECOGNITION_CONFIG = {
    'tolerance': 0.5,  # Порог схожести лиц
    'cooldown_time': 5  # Задержка между сохранениями (секунды)
}

# Пути к папкам
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
PHOTOS_FOLDER = "C:\\Users\\ACIIID\\Desktop\\project\\database\\recognized_humans"
FACES_FOLDER = "C:\\Users\\ACIIID\\Desktop\\project\\database\\recognized_faces"
DATABASE_PATH = "C:\\Users\\ACIIID\\Desktop\\project\\database\\faces_database"