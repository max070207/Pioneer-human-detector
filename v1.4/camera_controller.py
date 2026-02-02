from imports import *
from video_getter import VideoGetter

class CameraController:
    def __init__(self, file_manager: Any, log_maker: Any) -> None:
        self.file_manager: Any = file_manager
        self.log_maker: Any = log_maker
        self.logfile_name: str = self.file_manager.get_logfile_name()
        self.drone_connected: bool = False
        self.laptop_camera: Optional[Any] = None
        self.current_camera_type: str = "NONE"
        self.init_cameras()

    def init_cameras(self) -> None:
        """Инициализация камер в порядке приоритета: дрон -> ПК"""
        self.drone_connected = self.init_drone_camera()
        if self.drone_connected:
            self.current_camera_type = "DRONE"
            self.log_maker.writelog(self.logfile_name, 'Drone camera initialised.')
        else:
            self.drone_connected = False
            self.laptop_camera = self.init_laptop_camera()
            if self.laptop_camera is not None:
                self.current_camera_type = "PC"
                self.log_maker.writelog(self.logfile_name, 'PC camera initialised.')
            else:
                self.current_camera_type = "NONE"
                self.log_maker.writelog(self.logfile_name, 'Camera initialisation error.')

    def init_drone_camera(self) -> bool:
        """Инициализация камеры дрона"""
        try:
            from cam1 import Camera
            self.pioneer_cam = Camera()
            return True
        except ImportError as e:
            self.log_maker.writelog(self.logfile_name, f'Libraries input error:\n{e}')
            print(f"❌ Ошибка импорта библиотек дрона: {e}")
            return False
        except Exception as e:
            self.log_maker.writelog(self.logfile_name, f'Drone connection error:\n{e}')
            print(f"❌ Ошибка подключения к дрону: {e}")
            return False

    def init_laptop_camera(self) -> Optional[Any]:
        """Инициализация встроенной камеры ПК"""
        try:
            for camera_index in [0, 1, 2]:
                video_getter = VideoGetter(camera_index)
                if video_getter.isOpened():
                    video_getter.start()
                    time.sleep(0.5)
                    ret, frame = video_getter.read()
                    if ret and frame is not None:
                        self.log_maker.writelog(self.logfile_name, f'PC camera connected ({camera_index}).')
                        return video_getter
                    else:
                        video_getter.stop()
                else:
                    video_getter.stop()
            self.log_maker.writelog(self.logfile_name, 'PC camera not found.')
            return None
        except Exception as e:
            self.log_maker.writelog(self.logfile_name, f'PC camera initialisation error:\n{e}')
            print(f"❌ Ошибка инициализации встроенной камеры: {e}")
            return None

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Получение кадра с текущей камеры"""
        if self.current_camera_type == "DRONE":
            return self.get_drone_frame()
        elif self.current_camera_type == "PC":
            return self.get_laptop_frame()
        else:
            return self.get_simulation_frame()

    def get_drone_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Получение кадра с камеры дрона"""
        try:
            frame = self.pioneer_cam.get_cv_frame()
            if frame is not None and frame.size > 0:
                return True, frame
            else:
                self.log_maker.writelog(self.logfile_name, 'Drone camera unavainable, switching to PC camera...')
                self.switch_to_laptop_camera()
                return self.get_frame()
        except Exception as e:
            self.log_maker.writelog(self.logfile_name, f'Drone camera error, switching to PC camera...\n{e}')
            self.switch_to_laptop_camera()
            return self.get_frame()

    def get_laptop_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Получение кадра с встроенной камеры ноутбука"""
        try:
            if self.laptop_camera is not None:
                ret, frame = self.laptop_camera.read()
                if ret and frame is not None:
                    return True, frame
                else:
                    self.log_maker.writelog(self.logfile_name, 'Error while reading frame from PC camera.')
                    return False, None
            else:
                return self.get_simulation_frame()
        except Exception as e:
            self.log_maker.writelog(self.logfile_name, f'PC camera error:\n{e}')
            return self.get_simulation_frame()

    def get_simulation_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Создание тестового кадра если камеры недоступны"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "SIMULATION MODE - NO CAMERA", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to exit", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return True, frame

    def switch_to_laptop_camera(self) -> None:
        """Переключение на камеру ноутбука"""
        if self.current_camera_type == "DRONE":
            self.drone_connected = False
            self.current_camera_type = "PC"
            if self.laptop_camera is None:
                self.laptop_camera = self.init_laptop_camera()
            self.log_maker.writelog(self.logfile_name, 'Switching to PC camera...')

    def cleanup(self) -> None:
        """Очистка ресурсов всех камер"""
        if hasattr(self, 'pioneer_cam'):
            try:
                self.pioneer_cam.disconnect()
                self.log_maker.writelog(self.logfile_name, 'Drone disconnected.')
            except Exception as e:
                self.log_maker.writelog(self.logfile_name, f'Drone disconnection error:\n{e}')
        if self.laptop_camera is not None:
            if isinstance(self.laptop_camera, VideoGetter):
                self.laptop_camera.stop()
            else:
                self.laptop_camera.release()
            self.log_maker.writelog(self.logfile_name, 'PC camera disconnected.')
        self.log_maker.writelog(self.logfile_name, f'Camera resources have been released. {self.current_camera_type} camera was used.')