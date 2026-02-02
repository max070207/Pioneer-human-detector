# file_manager.py
from imports import *
from config import PHOTOS_FOLDER, FACES_FOLDER, LOGS_FOLDER

class FileManager:
    def __init__(self) -> None:
        self.photos_folder: str = self.create_folder(PHOTOS_FOLDER, "снимков")
        self.faces_folder: str = self.create_folder(FACES_FOLDER, "лиц")
        self.logs_folder: str = self.create_folder(LOGS_FOLDER, "логов")
        self.logs_file: str = self.create_file(LOGS_FOLDER)
        self.face_save_count: Dict[str, int] = {}

    def create_folder(self, folder_path: str, folder_type: str) -> str:
        """Создание папки если она не существует"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"✅ Создана папка для {folder_type}: {folder_path}")
        else:
            print(f"✅ Папка для {folder_type} уже существует: {folder_path}")
        return folder_path

    def create_file(self, folder_path: str) -> str:
        logfile = open(f"{folder_path}\\log_{str(datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')}.txt", 'w')
        print(f"Файл записей {logfile.name} создан")
        return logfile.name
    
    def get_logfile_name(self) -> str:
        return self.logs_file

    def _write_tmp_log(self, file: str, text: str) -> None:
        with open(file, 'a') as f:
            f.write(f"{datetime.now().replace(microsecond=0)} : {text}\n")

    def save_human_photo(self, frame: np.ndarray) -> bool:
        """Сохранение снимка при обнаружении человека (асинхронно)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"human_detected_{timestamp}.jpg"
            filepath = os.path.join(self.photos_folder, filename)
            threading.Thread(target=self._write_image_task, args=(filepath, frame.copy(), filename, "Human photo")).start()
            return True
        except Exception as e:
            self._write_tmp_log(self.logs_file, f'Error initiating save human photo:\n{e}')
            return False

    def _write_image_task(self, filepath: str, frame: np.ndarray, filename: str, log_prefix: str) -> None:
        try:
            cv2.imwrite(filepath, frame)
            self._write_tmp_log(self.logs_file, f'{log_prefix} saved: {filename}.')
        except Exception as e:
            self._write_tmp_log(self.logs_file, f'Error saving {filename}:\n{e}')

    def save_recognized_face(self, frame: np.ndarray, person_name: str) -> bool:
        """Сохранение снимка распознанного лица (асинхронно)"""
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
            threading.Thread(target=self._write_image_task, args=(filepath, frame.copy(), filename, "Recognized face photo")).start()
            return True
        except Exception as e:
            self._write_tmp_log(self.logs_file, f'Error initiating save {person_name} face:\n{e}')
            return False