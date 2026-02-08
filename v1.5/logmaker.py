from imports import *

class LogMaker:
    def __init__(self, file_manager: Any) -> None:
        self.file_manager: Any = file_manager
        self.init: None = self.writelog(self.file_manager.get_logfile_name(), 'Flight Initialisation.')

    def writelog(self, file: str, text: str) -> None:
        """Запись сообщения в лог-файл"""
        with open(file, 'a') as f:
            f.write(f"{datetime.now().replace(microsecond=0)} : {text}\n")