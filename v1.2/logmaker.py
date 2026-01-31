# logmaker.py
from imports import *
from file_manager import FileManager

class LogMaker:
    def __init__(self, file_manager) -> None:
        self.file_manager = file_manager
        self.init = self.writelog(self.file_manager.get_logfile_name(), 'Flight Initialisation.')

    def writelog(self, file, text) -> None:
        """Запись сообщения в лог-файл"""
        with open(file, 'a') as f:
            f.write(f"{datetime.now().replace(microsecond=0)} : {text}\n")