# main.py
from imports import *
from camera_controller import CameraController
from pose_detector import PoseDetector
from face_recognizer import FaceRecognizer
from file_manager import FileManager
from logmaker import LogMaker

class HumanDetector:
    def __init__(self) -> None:
        self.file_manager = FileManager()
        self.log_maker = LogMaker(self.file_manager)
        self.camera = CameraController(self.file_manager, self.log_maker)
        self.pose_detector = PoseDetector(self.file_manager, self.log_maker)
        self.face_recognizer = FaceRecognizer(self.file_manager, self.log_maker)
        self.previous_human_detected = False
        self.current_human_detected = False
        self.frame_count = 0
        self.fps = 0
        self.logfile_name = self.file_manager.get_logfile_name()
        
    def update_detection_status(self, human_detected, frame) -> None:
        """Обновление статуса обнаружения"""
        self.current_human_detected = human_detected
        if self.current_human_detected != self.previous_human_detected:
            if self.current_human_detected:
                self.log_maker.writelog(self.logfile_name, 'Human Found.')
                self.file_manager.save_human_photo(frame)
                self.log_maker.writelog(self.logfile_name, 'Face recognition activated.')
            else:
                self.log_maker.writelog(self.logfile_name, 'Human Lost.')
                self.log_maker.writelog(self.logfile_name, 'Face recognition deactivated.')
            self.previous_human_detected = self.current_human_detected

    def show_camera_feed(self) -> None:
        """Основная функция с поиском до первого сохранения"""
        fps_counter = 0
        last_fps_calc = time.time()
        while True:
            fps_counter += 1
            ret, raw_frame = self.camera.get_frame()
            if not ret:
                continue
            display_frame = raw_frame.copy()
            human_detected, display_frame = self.pose_detector.detect_and_draw_async(display_frame, self.frame_count)
            recognized_persons = self.face_recognizer.process_faces(raw_frame, self.frame_count, human_detected)
            display_frame = self.face_recognizer.draw_faces_and_message(display_frame, recognized_persons)
            if self.frame_count % 30 == 0:
                self.update_detection_status(human_detected, raw_frame)
            current_time = time.time()
            if current_time - last_fps_calc >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                last_fps_calc = current_time
            self.add_info_text(display_frame, human_detected)
            cv2.imshow('Pioneer-human-detector', display_frame)
            self.frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cleanup()

    def add_info_text(self, frame, human_detected) -> None:
        """Добавление информационного текста"""
        status_text = "HUMAN DETECTED" if human_detected else "HUMAN NOT DETECTED"
        color = (0, 255, 0) if human_detected else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"FPS: {self.fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.face_recognizer.face_search_active:
            search_status = "FACE SEARCH: ACTIVE"
            search_color = (255, 255, 0)
        elif self.face_recognizer.face_found:
            search_status = "FACE SEARCH: COMPLETED"
            search_color = (0, 255, 0)
        else:
            search_status = "FACE SEARCH: INACTIVE"
            search_color = (0, 0, 255)
        cv2.putText(frame, search_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, search_color, 1)

    def cleanup(self) -> None:
        """Очистка ресурсов"""
        self.camera.cleanup()
        self.pose_detector.cleanup()
        self.face_recognizer.cleanup()
        cv2.destroyAllWindows()
        print("✅ Ресурсы освобождены")

if __name__ == "__main__":
    detector = HumanDetector()
    detector.show_camera_feed()