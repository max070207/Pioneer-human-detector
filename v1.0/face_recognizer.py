# face_recognizer.py
from imports import *
from config import DATABASE_PATH, FACE_RECOGNITION_CONFIG, ASYNC_CONFIG

class FaceRecognizer:
    def __init__(self, file_manager, log_maker) -> None:
        self.file_manager = file_manager
        self.log_maker = log_maker
        self.logfile_name = self.file_manager.get_logfile_name()
        self.face_database = self.load_face_database()
        self.face_search_active = False
        self.face_found = False
        self.save_message_time = 0
        self.last_saved_face = None
        self.last_saved_face_location = None
        self.last_saved_face_similarity = 0
        self.processing_thread = None
        if ASYNC_CONFIG['face_processing']:
            self.is_processing = False
            self.latest_frame = None
            self.latest_results = []
            self.processing_lock = threading.Lock()

    def load_face_database(self):
        """Загрузка базы данных лиц"""
        face_database = {}
        if not os.path.exists(DATABASE_PATH):
            self.log_maker.writelog(self.logfile_name, 'Database folder was not found.')
            print(f"❌ Папка database не найдена: {DATABASE_PATH}")
            return face_database
        self.log_maker.writelog(self.logfile_name, 'Loading faces_database...')
        for filename in os.listdir(DATABASE_PATH):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(DATABASE_PATH, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        face_database[filename] = face_encodings[0]
                        self.log_maker.writelog(self.logfile_name, f'Loaded face: {filename}')
                except Exception as e:
                    self.log_maker.writelog(self.logfile_name, f'Error loading {filename}: {e}')
        self.log_maker.writelog(self.logfile_name, f'Successfully loaded faces_database. {len(face_database)} in total.')
        return face_database

    def start_face_search(self) -> None:
        """Запуск поиска лиц"""
        if not self.face_search_active:
            self.face_search_active = True
            self.face_found = False
            self.last_saved_face = None
            self.last_saved_face_location = None
            self.log_maker.writelog(self.logfile_name, 'Starting face recognition.')

    def stop_face_search(self) -> None:
        """Остановка поиска лиц"""
        self.face_search_active = False
        self.log_maker.writelog(self.logfile_name, 'Stopping face recognition.')

    def reset_face_search(self) -> None:
        """Полный сброс поиска лиц"""
        self.face_search_active = False
        self.face_found = False
        self.save_message_time = 0
        self.last_saved_face = None
        self.last_saved_face_location = None
        self.last_saved_face_similarity = 0

    def start_async_processing(self, frame, frame_count) -> None:
        """Запуск асинхронной обработки"""
        if not ASYNC_CONFIG['face_processing']or self.is_processing:
            return
        self.is_processing = True
        self.latest_frame = frame.copy()
        self.processing_thread = threading.Thread(
            target=self._process_faces_async,
            args=(self.latest_frame, frame_count),
            daemon=True
        )
        self.processing_thread.start()

    def _process_faces_async(self, frame, frame_count) -> None:
        """Асинхронная обработка лиц"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            recognized_persons = []
            if self.face_found:
                best_match_for_saved = None
                best_distance_for_saved = 0.7
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    for db_name, db_encoding in self.face_database.items():
                        if db_name == self.last_saved_face:
                            face_distance = face_recognition.face_distance([db_encoding], face_encoding)[0]
                            if face_distance < best_distance_for_saved:
                                best_distance_for_saved = face_distance
                                best_match_for_saved = (top, right, bottom, left, face_distance)
                                break
                if best_match_for_saved:
                    top, right, bottom, left, face_distance = best_match_for_saved
                    self.last_saved_face_location = (top, right, bottom, left)
                    self.last_saved_face_similarity = (1 - face_distance) * 100
            if not self.face_found:
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    person_name = "Unknown"
                    best_match_distance = 1.0
                    for db_name, db_encoding in self.face_database.items():
                        face_distance = face_recognition.face_distance([db_encoding], face_encoding)[0]
                        if face_distance < FACE_RECOGNITION_CONFIG['tolerance'] and face_distance < best_match_distance:
                            best_match_distance = face_distance
                            person_name = db_name
                    similarity_percent = (1 - best_match_distance) * 100 if person_name != "Unknown" else 0.0
                    if not self.face_found and person_name != "Unknown":
                        self.log_maker.writelog(self.logfile_name, f'Saved face: {person_name} (similarity: {similarity_percent:.1f}%).')
                        self.file_manager.save_recognized_face(frame, person_name)
                        self.face_found = True
                        self.last_saved_face = person_name
                        self.last_saved_face_location = (top, right, bottom, left)
                        self.last_saved_face_similarity = similarity_percent
                        self.save_message_time = time.time()
                        self.face_search_active = False
                        break
                    recognized_persons.append((person_name, (top, right, bottom, left), similarity_percent))
            with self.processing_lock:
                self.latest_results = recognized_persons
        except Exception as e:
            self.log_maker.writelog(self.logfile_name, f'Face processing error: {e}.')
            with self.processing_lock:
                self.latest_results = []
        finally:
            self.is_processing = False

    def get_latest_results(self):
        """Получение последних результатов обработки"""
        if not ASYNC_CONFIG['face_processing']:
            return []
        with self.processing_lock:
            return self.latest_results.copy()

    def process_faces(self, raw_frame, frame_count, is_human_detected):
        """Обработка лиц - оптимизированная для отслеживания"""
        if not is_human_detected:
            if self.face_found or self.face_search_active:
                self.reset_face_search()
            return []
        processing_interval = 1 if self.face_found else 2
        if frame_count % processing_interval == 0:
            self.start_async_processing(raw_frame, frame_count)
        if is_human_detected and not self.face_search_active and not self.face_found:
            self.start_face_search()
        return self.get_latest_results()

    def draw_faces_and_message(self, frame, recognized_persons):
        """Отрисовка лиц и сообщения о сохранении БЕЗ отладочных сообщений"""
        if self.face_search_active and not self.face_found and recognized_persons:
            for person_name, (top, right, bottom, left), similarity_percent in recognized_persons:
                if self._is_valid_face_coordinates(left, top, right, bottom, frame):
                    color = (255, 255, 0) if person_name != "Unknown" else (0, 0, 255)
                    self._draw_face_rectangle(frame, left, top, right, bottom, color)
        is_tracking_period = self.face_found and time.time() - self.save_message_time < 5
        if is_tracking_period:
            if self.last_saved_face_location:
                top, right, bottom, left = self.last_saved_face_location
                if self._is_valid_face_coordinates(left, top, right, bottom, frame):
                    color = (0, 255, 0)
                    self._draw_face_rectangle(frame, left, top, right, bottom, color)
            self._draw_save_message(frame)
        elif self.face_found and time.time() - self.save_message_time >= 5:
            self.last_saved_face_location = None
        return frame

    def _is_valid_face_coordinates(self, left, top, right, bottom, frame) -> bool:
        """Проверка валидности координат лица"""
        frame_height, frame_width = frame.shape[:2]
        if (left >= right or top >= bottom or left < 0 or top < 0 or right > frame_width or bottom > frame_height):
            return False
        face_width = right - left
        face_height = bottom - top
        if face_width < 30 or face_height < 30:
            return False
        return True
    
    def _draw_face_rectangle(self, frame, left, top, right, bottom, color) -> None:
        """Отрисовка прямоугольника вокруг лица"""
        thickness = 2
        distance = (abs(top-bottom))
        lines = [((left, top), (left+distance//5, top)), ((right, top), (right-distance//5, top)),
                ((right, top), (right, top+distance//5)), ((right, bottom), (right, bottom-distance//5)),
                ((right, bottom), (right-distance//5, bottom)), ((left, bottom), (left+distance//5, bottom)),
                ((left, bottom), (left, bottom-distance//5)), ((left, top), (left, top+distance//5))]
        for line in lines:
            cv2.line(frame, line[0], line[1], color, thickness)

    def _draw_save_message(self, frame) -> None:
        """Отрисовка сообщения о сохранении лица"""
        frame_height = frame.shape[0]
        message = f"FOUND FACE: {os.path.splitext(self.last_saved_face)[0]}"
        cv2.putText(frame, message, (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def cleanup(self) -> None:
        """Очистка ресурсов"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)