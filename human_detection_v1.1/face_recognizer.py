# face_recognizer.py
from imports import *
from config import DATABASE_PATH, FACE_RECOGNITION_CONFIG, ASYNC_CONFIG

class FaceRecognizer:
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.face_database = self.load_face_database()
        self.saved_faces = set()  # Уже сохраненные лица
        
        # Асинхронная обработка
        if ASYNC_CONFIG['face_processing']:
            self.processing_thread = None
            self.is_processing = False
            self.latest_frame = None
            self.latest_results = []
            self.processing_lock = threading.Lock()
        else:
            self.processing_thread = None
            
        self.last_processed_frame = 0
        self.processing_interval = 2

    def load_face_database(self):
        """Загрузка базы данных лиц"""
        face_database = {}
        if not os.path.exists(DATABASE_PATH):
            print(f"❌ Папка database не найдена: {DATABASE_PATH}")
            return face_database
        
        print("🔄 Загрузка базы данных лиц...")
        for filename in os.listdir(DATABASE_PATH):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(DATABASE_PATH, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        face_database[filename] = face_encodings[0]
                        print(f"✅ Загружено лицо: {filename}")
                except Exception as e:
                    print(f"❌ Ошибка загрузки {filename}: {e}")
        
        print(f"✅ База данных загружена. Лиц в базе: {len(face_database)}")
        return face_database

    def start_async_processing(self, frame, frame_count):
        """Запуск асинхронной обработки для поиска лиц"""
        if not ASYNC_CONFIG['face_processing']:
            return
            
        if frame_count - self.last_processed_frame < self.processing_interval:
            return
            
        if self.is_processing:
            return
            
        self.is_processing = True
        self.latest_frame = frame.copy()
        self.last_processed_frame = frame_count
        
        self.processing_thread = threading.Thread(
            target=self._process_faces_async,
            args=(self.latest_frame, frame_count),
            daemon=True
        )
        self.processing_thread.start()

    def _process_faces_async(self, frame, frame_count):
        """Асинхронная обработка лиц с сохранением при первом обнаружении"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(
                rgb_frame, 
                model="hog",
                number_of_times_to_upsample=0
            )
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            recognized_persons = []
            new_faces_found = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                person_name = "Unknown"
                best_match_distance = 1.0
                
                for db_name, db_encoding in self.face_database.items():
                    face_distance = face_recognition.face_distance([db_encoding], face_encoding)[0]
                    if face_distance < FACE_RECOGNITION_CONFIG['tolerance'] and face_distance < best_match_distance:
                        best_match_distance = face_distance
                        person_name = db_name
                
                if person_name != "Unknown":
                    similarity_percent = (1 - best_match_distance) * 100
                    
                    if person_name not in self.saved_faces:
                        print(f"💾 СОХРАНЯЮ НОВОЕ ЛИЦО: {person_name} (схожесть: {similarity_percent:.1f}%)")
                        self.file_manager.save_recognized_face(frame, person_name)
                        self.saved_faces.add(person_name)
                        new_faces_found.append(person_name)
                        
                else:
                    similarity_percent = 0.0
                
                recognized_persons.append((person_name, (top, right, bottom, left), similarity_percent))
            
            if new_faces_found:
                print(f"✅ Сохранены новые лица: {', '.join(new_faces_found)}")
                print(f"📊 Всего сохраненных лиц: {len(self.saved_faces)}")
            
            with self.processing_lock:
                self.latest_results = recognized_persons
                
        except Exception as e:
            print(f"Face processing error: {e}")
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
        """Обработка лиц"""
        if is_human_detected:
            self.start_async_processing(raw_frame, frame_count)
        
        return self.get_latest_results()

    def draw_faces_smooth(self, frame, recognized_persons):
        """Плавная отрисовка лиц БЕЗ эмодзи и с правильными именами"""
        if not recognized_persons:
            return frame
            
        frame_height, frame_width = frame.shape[:2]
        
        for person_name, (top, right, bottom, left), similarity_percent in recognized_persons:
            # Проверяем валидность координат
            if (left >= right or top >= bottom or 
                left < 0 or top < 0 or 
                right > frame_width or bottom > frame_height):
                continue
                
            # Проверяем размер лица
            face_width = right - left
            face_height = bottom - top
            if face_width < 30 or face_height < 30:
                continue
                
            if person_name != "Unknown":
                # Убираем расширение файла из имени
                display_name = os.path.splitext(person_name)[0]
                
                if person_name in self.saved_faces:
                    color = (0, 255, 0)  # Зеленый - уже сохранено
                    # БЕЗ символов - просто имя и процент
                    label = f"{display_name} ({similarity_percent:.1f}%)"
                else:
                    color = (255, 255, 0)  # Желтый - новое лицо
                    # БЕЗ символов - просто имя и процент
                    label = f"{display_name} ({similarity_percent:.1f}%)"
            else:
                color = (0, 0, 255)  # Красный - неизвестное
                label = "Unknown"
            
            # Рисуем прямоугольник
            thickness = 2
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            
            # Фон для текста
            text_bg_top = max(0, top - 25)
            text_bg_bottom = top
            cv2.rectangle(frame, (left, text_bg_top), (right, text_bg_bottom), color, cv2.FILLED)
            
            # Текст БЕЗ эмодзи
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            
            # Сокращаем длинные имена (если нужно)
            text_width = cv2.getTextSize(label, font, font_scale, 1)[0][0]
            available_width = right - left - 10
            
            if text_width > available_width and person_name != "Unknown":
                # Сокращаем имя если не помещается
                max_chars = min(8, len(display_name))
                shortened_name = display_name[:max_chars]
                label = f"{shortened_name} ({similarity_percent:.1f}%)"
            
            text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
            text_x = left + (right - left - text_size[0]) // 2
            text_y = top - 8
            
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), 1)
        
        return frame

    def get_saved_faces_count(self):
        """Получение количества сохраненных лиц"""
        return len(self.saved_faces)

    def cleanup(self):
        """Очистка ресурсов"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)