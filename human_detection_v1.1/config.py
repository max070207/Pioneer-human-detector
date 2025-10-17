# config.py
from imports import *

# Базовые настройки
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'min_tracking_confidence': 0.5,
    'min_detection_confidence': 0.5,
    'model_complexity': 1
}

FACE_RECOGNITION_CONFIG = {
    'tolerance': 0.5,
    'cooldown_time': 5
}

# Настройки асинхронной обработки
ASYNC_CONFIG = {
    'pose_processing': True,      # Включить асинхронную обработку скелета
    'face_processing': True,      # Включить асинхронную обработку лиц
    'max_queue_size': 2,          # Максимальный размер очереди кадров
    'processing_timeout': 2.0     # Таймаут обработки в секундах
}

# Пути к папкам
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
PHOTOS_FOLDER = "C:\\Users\\ACIIID\\Desktop\\project\\database\\recognized_humans"
FACES_FOLDER = "C:\\Users\\ACIIID\\Desktop\\project\\database\\recognized_faces"
DATABASE_PATH = "C:\\Users\\ACIIID\\Desktop\\project\\database\\faces_database"