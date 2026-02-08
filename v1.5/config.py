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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FOLDER = os.path.join(BASE_DIR, "database")
PHOTOS_FOLDER = os.path.join(DATABASE_FOLDER, "recognized_humans")
FACES_FOLDER = os.path.join(DATABASE_FOLDER, "recognized_faces")
LOGS_FOLDER = os.path.join(DATABASE_FOLDER, "logs")
DATABASE_PATH = os.path.join(DATABASE_FOLDER, "faces_database")