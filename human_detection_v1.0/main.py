# main.py
import cv2
import numpy as np
import time
from camera_controller import CameraController
from pose_detector import PoseDetector
from face_recognizer import FaceRecognizer
from file_manager import FileManager

class HumanDetector:
    def __init__(self):
        self.file_manager = FileManager()
        self.camera = CameraController()
        self.pose_detector = PoseDetector()
        self.face_recognizer = FaceRecognizer(self.file_manager)
        
        # Состояние обнаружения
        self.previous_human_detected = False
        self.current_human_detected = False
        self.face_tracking_active = False  # Флаг активного отслеживания лиц
        self.tracked_faces = set()  # Множество уже отслеживаемых лиц

    def update_detection_status(self, human_detected, frame):
        """Обновление статуса обнаружения"""
        self.current_human_detected = human_detected

        if self.current_human_detected != self.previous_human_detected:
            if self.current_human_detected:
                print("✅ человек обнаружен")
                self.file_manager.save_human_photo(frame)
                # Активируем отслеживание лиц после обнаружения человека
                self.face_tracking_active = True
                print("👤 Активировано отслеживание лиц")
            else:
                print("❌ человек потерян")
                # Деактивируем отслеживание лиц
                self.face_tracking_active = False
                self.tracked_faces.clear()  # Очищаем список отслеживаемых лиц
                print("👤 Деактивировано отслеживание лиц")

            self.previous_human_detected = self.current_human_detected

    def show_camera_feed(self):
        """Основная функция для отображения окна с камерой дрона"""
        if not self.camera.drone_connected:
            print("⚠️  Работа в режиме симуляции (дрон не подключен)")

        print("🚁 Запуск системы обнаружения человека с камеры дрона...")
        print("👤 Отслеживание лиц активируется после обнаружения человека")
        print("Для выхода нажмите 'q' в окне видео")

        frame_count = 0

        while True:
            ret, raw_frame = self.camera.get_frame()

            if not ret:
                print(f"❌ Ошибка получения кадра {frame_count}")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "DRONE CAMERA ERROR - NO VIDEO FEED", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # КОПИЯ кадра для отображения (будет с скелетом и лицами)
                display_frame = raw_frame.copy()

                # 1. ОБНАРУЖЕНИЕ СКЕЛЕТА И ОТРИСОВКА
                human_detected_in_frame, display_frame = self.pose_detector.detect_and_draw(display_frame)

                # 2. ОТСЛЕЖИВАНИЕ ЛИЦ (только если обнаружен человек)
                recognized_persons = []
                if self.face_tracking_active:
                    # Используем чистый кадр для распознавания лиц
                    recognized_persons = self.face_recognizer.detect_and_track_faces(
                        raw_frame, 
                        self.tracked_faces
                    )
                    
                    # 3. ОТРИСОВКА ЛИЦ
                    display_frame = self.face_recognizer.draw_faces(display_frame, recognized_persons)

                # 4. Анализ состояния для вывода в консоль
                if frame_count % 30 == 0:
                    self.update_detection_status(human_detected_in_frame, raw_frame)

                # Добавление информационного текста
                self.add_info_text(display_frame, human_detected_in_frame, frame_count, recognized_persons)

                frame = display_frame

            window_name = '🚁 Обнаружение человека и отслеживание лиц - Камера дрона'
            cv2.imshow(window_name, frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def add_info_text(self, frame, human_detected, frame_count, recognized_persons):
        """Добавление информационного текста на кадр"""
        status_text = "HUMAN DETECTED" if human_detected else "HUMAN NOT DETECTED"
        color = (0, 255, 0) if human_detected else (0, 0, 255)

        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Статус отслеживания лиц
        tracking_status = "FACE TRACKING: ACTIVE" if self.face_tracking_active else "FACE TRACKING: INACTIVE"
        tracking_color = (0, 255, 0) if self.face_tracking_active else (0, 0, 255)
        cv2.putText(frame, tracking_status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_color, 2)

        # Информация о камере
        if self.camera.current_camera_type == "DRONE":
            camera_status = "DRONE CAMERA - CONNECTED"
            camera_color = (0, 255, 0)
        elif self.camera.current_camera_type == "LAPTOP":
            camera_status = "LAPTOP CAMERA - ACTIVE"
            camera_color = (0, 255, 255)  # Желтый для камеры ноутбука
        else:
            camera_status = "SIMULATION MODE"
            camera_color = (0, 165, 255)  # Оранжевый для симуляции

        cv2.putText(frame, f"CAMERA: {camera_status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, camera_color, 2)

        cv2.putText(frame, f"FRAME: {frame_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Статистика распознавания
        if self.face_tracking_active:
            recognized_count = len([p for p in recognized_persons if p[0] != "Unknown"])
            faces_text = f"TRACKED FACES: {len(recognized_persons)} (KNOWN: {recognized_count})"
            cv2.putText(frame, faces_text, (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            tracked_count = f"UNIQUE FACES: {len(self.tracked_faces)}"
            cv2.putText(frame, tracked_count, (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def cleanup(self):
        """Очистка ресурсов"""
        self.camera.cleanup()
        self.pose_detector.cleanup()
        cv2.destroyAllWindows()
        print("✅ Ресурсы освобождены")
        print(f"📁 Снимки сохранены в: {self.file_manager.photos_folder}")
        print(f"📁 Распознанные лица сохранены в: {self.file_manager.faces_folder}")
        print(f"👤 Уникальных лиц отслежено: {len(self.tracked_faces)}")


# Запуск системы
if __name__ == "__main__":
    detector = HumanDetector()
    detector.show_camera_feed()