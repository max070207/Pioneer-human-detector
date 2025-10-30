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
    'cooldown_time': 5
}

ASYNC_CONFIG = {
    'pose_processing': True,
    'face_processing': True,
    'max_queue_size': 2,
    'processing_timeout': 2.0
}

DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
PHOTOS_FOLDER = "C:\\Users\\ACIIID\\Desktop\\project\\database\\recognized_humans"
FACES_FOLDER = "C:\\Users\\ACIIID\\Desktop\\project\\database\\recognized_faces"
DATABASE_PATH = "C:\\Users\\ACIIID\\Desktop\\project\\database\\faces_database"
LOGS_FOLDER = "C:\\Users\\ACIIID\\Desktop\\project\\database\\logs"