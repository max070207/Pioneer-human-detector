# file_manager.py
from imports import *
from config import PHOTOS_FOLDER, FACES_FOLDER

class FileManager:
    def __init__(self):
        self.photos_folder = self.create_folder(PHOTOS_FOLDER, "снимков")
        self.faces_folder = self.create_folder(FACES_FOLDER, "лиц")
        self.face_save_count = {}
        self.recognition_cooldown = {}

    def create_folder(self, folder_path, folder_type):
        """Создание папки если она не существует"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"✅ Создана папка для {folder_type}: {folder_path}")
        else:
            print(f"✅ Папка для {folder_type} уже существует: {folder_path}")
        return folder_path

    def save_human_photo(self, frame):
        """Сохранение снимка при обнаружении человека"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"human_detected_{timestamp}.jpg"
            filepath = os.path.join(self.photos_folder, filename)

            cv2.imwrite(filepath, frame)
            print(f"📸 Снимок сохранен: {filepath}")

            return True
        except Exception as e:
            print(f"❌ Ошибка при сохранении снимка: {e}")
            return False

    def save_recognized_face(self, frame, person_name):
        """Сохранение снимка распознанного лица"""
        try:
            if not isinstance(person_name, str):
                person_name = str(person_name)
            
            if person_name not in self.face_save_count:
                self.face_save_count[person_name] = 0
            
            self.face_save_count[person_name] += 1
            count = self.face_save_count[person_name]
            
            base_name = os.path.splitext(person_name)[0]
            filename = f"{base_name}_{count}.jpg"
            filepath = os.path.join(self.faces_folder, filename)

            cv2.imwrite(filepath, frame)
            print(f"📸 Распознанное лицо сохранено: {filename}")

            return True
        except Exception as e:
            print(f"❌ Ошибка при сохранении распознанного лица '{person_name}': {e}")
            return False