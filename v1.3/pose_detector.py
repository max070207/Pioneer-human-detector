# pose_detector.py
from imports import *
from config import MEDIAPIPE_CONFIG, ASYNC_CONFIG
import multiprocessing
from mediapipe.framework.formats import landmark_pb2

logging.getLogger('mediapipe').setLevel(logging.ERROR)

def pose_worker(input_queue, output_queue, config):
    """Worker function for pose detection process"""
    import mediapipe as mp
    import cv2
    
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(**config)
    
    while True:
        try:
            task = input_queue.get()
            if task is None:
                break
            
            frame, frame_count = task
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb_frame)
            
            landmarks_data = None
            if results.pose_landmarks:
                landmarks_data = []
                for lm in results.pose_landmarks.landmark:
                    landmarks_data.append({
                        'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility
                    })
            
            output_queue.put((landmarks_data, frame_count))
        except Exception as e:
            # print(f"Pose worker error: {e}")
            continue

class PoseDetector:
    def __init__(self, file_manager, log_maker) -> None:
        self.file_manager = file_manager
        self.log_maker = log_maker
        self.logfile_name = self.file_manager.get_logfile_name()
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.input_queue = multiprocessing.Queue(maxsize=1)
        self.output_queue = multiprocessing.Queue(maxsize=1)
        self.process = None
        self.last_landmarks = None
        
        if ASYNC_CONFIG['pose_processing']:
            self.start_process()
        else:
            self.pose_detector = self.mp_pose.Pose(**MEDIAPIPE_CONFIG)

    def start_process(self):
        self.process = multiprocessing.Process(
            target=pose_worker,
            args=(self.input_queue, self.output_queue, MEDIAPIPE_CONFIG),
            daemon=True
        )
        self.process.start()

    def detect_and_draw_async(self, frame, frame_count):
        """Асинхронное обнаружение и отрисовка"""
        human_detected = False
        
        if self.process:
            # Send frame if queue is empty (drop frame)
            try:
                self.input_queue.put_nowait((frame.copy(), frame_count))
            except queue.Full:
                pass
                
            # Get latest result
            try:
                while not self.output_queue.empty():
                    landmarks_data, _ = self.output_queue.get_nowait()
                    if landmarks_data:
                        # Reconstruct landmarks for drawing
                        landmark_list = landmark_pb2.NormalizedLandmarkList()
                        for lm_dict in landmarks_data:
                            lm = landmark_list.landmark.add()
                            lm.x = lm_dict['x']
                            lm.y = lm_dict['y']
                            lm.z = lm_dict['z']
                            lm.visibility = lm_dict['visibility']
                        self.last_landmarks = landmark_list
                    else:
                        self.last_landmarks = None
            except queue.Empty:
                pass
        else:
            # Fallback synchronous
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_frame)
            self.last_landmarks = results.pose_landmarks

        if self.last_landmarks:
            human_detected = True
            self.mp_drawing.draw_landmarks(
                frame,
                self.last_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
        return human_detected, frame

    def cleanup(self) -> None:
        """Очистка ресурсов"""
        if self.process:
            self.input_queue.put(None)
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()
        elif hasattr(self, 'pose_detector'):
            self.pose_detector.close()