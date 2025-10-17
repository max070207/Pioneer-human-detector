# face_recognizer.py
import cv2
import face_recognition
import os
from config import DATABASE_PATH, FACE_RECOGNITION_CONFIG

class FaceRecognizer:
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.face_database = self.load_face_database()

    def load_face_database(self):
        """Загрузка базы данных лиц из папки database"""
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
                    else:
                        print(f"⚠️  Лицо не найдено на изображении: {filename}")
                        
                except Exception as e:
                    print(f"❌ Ошибка загрузки {filename}: {e}")
        
        print(f"✅ База данных загружена. Лиц в базе: {len(face_database)}")
        return face_database

    def detect_and_track_faces(self, raw_frame, tracked_faces):
        """
        Распознавание и отслеживание лиц на кадре
        Возвращает распознанных лиц и сохраняет новые уникальные лица
        """
        rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_persons = []
        new_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            person_name = "Unknown"
            best_match_distance = 1.0
            similarity_percent = 100
            
            # Сравниваем с базой данных
            for db_name, db_encoding in self.face_database.items():
                face_distance = face_recognition.face_distance([db_encoding], face_encoding)[0]
                
                if face_distance < FACE_RECOGNITION_CONFIG['tolerance'] and face_distance < best_match_distance:
                    best_match_distance = face_distance
                    person_name = db_name
                    similarity_percent = (1 - best_match_distance) * 100
            
            recognized_persons.append((person_name, (top, right, bottom, left), similarity_percent))
            
            # Сохраняем только новые уникальные лица
            if person_name != "Unknown" and person_name not in tracked_faces:
                print(f"👤 Новое лицо обнаружено: {person_name} (схожесть: {similarity_percent:.1f}/100)")
                self.file_manager.save_recognized_face(raw_frame, person_name)
                tracked_faces.add(person_name)  # Добавляем в отслеживаемые
                new_faces.append(person_name)
        
        # Вывод информации о новых лицах
        if new_faces:
            print(f"✅ Добавлены в отслеживание: {', '.join(new_faces)}")
        
        return recognized_persons

    def draw_faces(self, frame, recognized_persons):
        """Отрисовка лиц на кадре"""
        for person_name, (top, right, bottom, left), distance in recognized_persons:
            # Разные цвета для известных и неизвестных лиц
            if person_name != "Unknown":
                color = (0, 255, 0)  # Зеленый для известных
                label = f"{person_name} ({distance:.1f})"
            else:
                color = (0, 0, 255)  # Красный для неизвестных
                label = "Unknown"
            
            # Рисуем прямоугольник вокруг лица
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Добавляем подпись
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.4, (255, 255, 255), 1)
        
        return frame