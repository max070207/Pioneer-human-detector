# pose_detector.py
from imports import *
from config import MEDIAPIPE_CONFIG

# Настройка логирования для уменьшения вывода MediaPipe
logging.getLogger('mediapipe').setLevel(logging.ERROR)

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_detector = self.mp_pose.Pose(**MEDIAPIPE_CONFIG)

    def detect_and_draw(self, frame):
        """
        Обнаружение и отрисовка человека в кадре
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb_frame)

        human_detected = False

        if results.pose_landmarks is not None:
            human_detected = True
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        return human_detected, frame

    def cleanup(self):
        """Очистка ресурсов"""
        self.pose_detector.close()