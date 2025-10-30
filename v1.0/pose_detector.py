# pose_detector.py
from imports import *
from config import MEDIAPIPE_CONFIG, ASYNC_CONFIG
from async_processor import AsyncProcessor

logging.getLogger('mediapipe').setLevel(logging.ERROR)

class PoseDetector:
    def __init__(self, file_manager, log_maker) -> None:
        self.file_manager = file_manager
        self.log_maker = log_maker
        self.logfile_name = self.file_manager.get_logfile_name()
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_detector = self.mp_pose.Pose(**MEDIAPIPE_CONFIG)
        if ASYNC_CONFIG['pose_processing']:
            self.async_processor = AsyncProcessor(
                processing_function=self._process_pose_sync,
                max_queue_size=ASYNC_CONFIG['max_queue_size'],
                timeout=ASYNC_CONFIG['processing_timeout']
            )
            self.async_processor.start()
        else:
            self.async_processor = None
        self.last_results = None

    def _process_pose_sync(self, frame_data):
        """Синхронная обработка позы (вызывается в отдельном потоке)"""
        try:
            frame, frame_count = frame_data
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_frame)
            return (results, frame_count)
        except Exception as e:
            self.log_maker.writelog(self.logfile_name, f'Pose processing error:\n{e}')
            print(f"Pose processing error: {e}")
            return (None, frame_data[1] if frame_data else 0)

    def detect_and_draw_async(self, frame, frame_count):
        """Асинхронное обнаружение и отрисовка"""
        human_detected = False
        if self.async_processor:
            result = self.async_processor.process_async((frame, frame_count))
            if result:
                results, processed_frame_count = result
                self.last_results = results
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_results = self.pose_detector.process(rgb_frame)
        if self.last_results and self.last_results.pose_landmarks is not None:
            human_detected = True
            self.mp_drawing.draw_landmarks(
                frame,
                self.last_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        return human_detected, frame

    def cleanup(self) -> None:
        """Очистка ресурсов"""
        if self.async_processor:
            self.async_processor.stop()
        self.pose_detector.close()