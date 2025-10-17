# main.py
from imports import *
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
        self.face_tracking_active = False
        self.tracked_faces = set()
        
        # FPS счетчик
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0

    def update_detection_status(self, human_detected, frame):
        """Обновление статуса обнаружения"""
        self.current_human_detected = human_detected

        if self.current_human_detected != self.previous_human_detected:
            if self.current_human_detected:
                print("✅ человек обнаружен")
                self.file_manager.save_human_photo(frame)
                self.face_tracking_active = True
                print("👤 Активировано отслеживание лиц")
            else:
                print("❌ человек потерян")
                self.face_tracking_active = False
                self.tracked_faces.clear()
                print("👤 Деактивировано отслеживание лиц")

            self.previous_human_detected = self.current_human_detected

    def show_camera_feed(self):
        """Основная функция с сохранением лиц при первом обнаружении"""
        print("🚁 Запуск системы с сохранением лиц при первом обнаружении...")
        print("💾 Каждое новое лицо сохраняется один раз при первом нахождении")
        
        fps_counter = 0
        last_fps_calc = time.time()

        while True:
            frame_start = time.time()
            fps_counter += 1
            
            # Получение кадра
            ret, raw_frame = self.camera.get_frame()

            if not ret:
                continue
            
            display_frame = raw_frame.copy()

            # 1. АСИНХРОННОЕ ОБНАРУЖЕНИЕ СКЕЛЕТА
            human_detected_in_frame, display_frame = self.pose_detector.detect_and_draw_async(
                display_frame, self.frame_count
            )

            # 2. ПОСТОЯННЫЙ ПОИСК ЛИЦ с сохранением при первом обнаружении
            recognized_persons = self.face_recognizer.process_faces(
                raw_frame, self.frame_count, human_detected_in_frame
            )
            
            # Отрисовываем лица (даже если человека нет, но есть сохраненные результаты)
            display_frame = self.face_recognizer.draw_faces_smooth(display_frame, recognized_persons)

            # 3. Обновление статуса
            if self.frame_count % 45 == 0:
                self.update_detection_status(human_detected_in_frame, raw_frame)

            # 4. Расчет FPS
            current_time = time.time()
            if current_time - last_fps_calc >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                last_fps_calc = current_time

            # 5. Информация на экране
            self.add_info_text(display_frame, human_detected_in_frame, recognized_persons)

            # 6. Отображение
            cv2.imshow('🚁 Сохранение лиц при первом обнаружении', display_frame)

            self.frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def add_info_text(self, frame, human_detected, recognized_persons):
        """Добавление информационного текста БЕЗ легенды цветов"""
        status_text = "HUMAN DETECTED" if human_detected else "HUMAN NOT DETECTED"
        color = (0, 255, 0) if human_detected else (0, 0, 255)

        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"FPS: {self.fps}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Статистика лиц (только цифры)
        saved_count = self.face_recognizer.get_saved_faces_count()
        current_known = len([p for p in recognized_persons if p[0] != "Unknown"])
        
        saved_text = f"SAVED: {saved_count} | CURRENT: {current_known}"
        cv2.putText(frame, saved_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # УБИРАЕМ легенду цветов полностью

    def cleanup(self):
        """Очистка ресурсов"""
        self.camera.cleanup()
        self.pose_detector.cleanup()
        self.face_recognizer.cleanup()
        cv2.destroyAllWindows()
        print("✅ Ресурсы освобождены")

if __name__ == "__main__":
    detector = HumanDetector()
    detector.show_camera_feed()