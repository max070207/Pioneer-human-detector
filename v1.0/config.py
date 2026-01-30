# config.py
from imports import *

MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'min_tracking_confidence': 0.5,
    'min_detection_confidence': 0.5,
    'model_complexity': 1
}

FACE_RECOGNITION_CONFIG = {
    'tolerance': 0.5,
    'cooldown_time': 5,
    'model': 'hog' # 'hog' (faster, CPU) or 'cnn' (slower, GPU/CUDA required)
}

ASYNC_CONFIG = {
    'pose_processing': True,
    'face_processing': True,
    'max_queue_size': 2,
    'processing_timeout': 2.0
}

DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
PHOTOS_FOLDER = os.path.join(DESKTOP_PATH, "recognized_humans")
FACES_FOLDER = os.path.join(DESKTOP_PATH, "recognized_faces")
DATABASE_PATH = os.path.join(DESKTOP_PATH, "faces_database")
LOGS_FOLDER = os.path.join(DESKTOP_PATH, "logs")
