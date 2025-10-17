# camera_controller.py
import cv2
import numpy as np

class CameraController:
    def __init__(self):
        self.drone_connected = False
        self.laptop_camera = None
        self.current_camera_type = "NONE"
        self.init_cameras()

    def init_cameras(self):
        """Инициализация камер в порядке приоритета: дрон -> ноутбук"""
        # Сначала пробуем подключиться к дрону
        self.drone_connected = self.init_drone_camera()
        
        if self.drone_connected:
            self.current_camera_type = "DRONE"
            print("✅ Камера дрона инициализирована")
        else:
            # Если дрон недоступен, пробуем встроенную камеру ноутбука
            self.drone_connected = False
            self.laptop_camera = self.init_laptop_camera()
            
            if self.laptop_camera is not None:
                self.current_camera_type = "LAPTOP"
                print("✅ Встроенная камера ноутбука инициализирована")
            else:
                self.current_camera_type = "NONE"
                print("❌ Ни одна камера не доступна")

    def init_drone_camera(self):
        """Инициализация камеры дрона"""
        try:
            from pioneer_sdk import Pioneer
            from cam1 import Camera

            self.pioneer = Pioneer(logger=True, log_connection=True)
            self.pioneer_cam = Camera()
            return True

        except ImportError as e:
            print(f"❌ Ошибка импорта библиотек дрона: {e}")
            return False
        except Exception as e:
            print(f"❌ Ошибка подключения к дрону: {e}")
            return False

    def init_laptop_camera(self):
        """Инициализация встроенной камеры ноутбука"""
        try:
            # Пробуем разные индексы камер (0, 1, 2...)
            for camera_index in [0, 1, 2]:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    # Проверяем, что камера действительно работает
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Найдена встроенная камера (индекс {camera_index})")
                        return cap
                    else:
                        cap.release()
                else:
                    cap.release()
            
            print("❌ Встроенная камера ноутбука не найдена")
            return None
            
        except Exception as e:
            print(f"❌ Ошибка инициализации встроенной камеры: {e}")
            return None

    def get_frame(self):
        """Получение кадра с текущей камеры"""
        if self.current_camera_type == "DRONE":
            return self.get_drone_frame()
        elif self.current_camera_type == "LAPTOP":
            return self.get_laptop_frame()
        else:
            return self.get_simulation_frame()

    def get_drone_frame(self):
        """Получение кадра с камеры дрона"""
        try:
            frame = self.pioneer_cam.get_cv_frame()
            if frame is not None and frame.size > 0:
                return True, frame
            else:
                print("⚠️  Камера дрона недоступна, переключаюсь на ноутбук...")
                self.switch_to_laptop_camera()
                return self.get_frame()  # Рекурсивно пробуем получить кадр с новой камеры
        except Exception as e:
            print(f"⚠️  Ошибка камеры дрона: {e}, переключаюсь на ноутбук...")
            self.switch_to_laptop_camera()
            return self.get_frame()

    def get_laptop_frame(self):
        """Получение кадра с встроенной камеры ноутбука"""
        try:
            if self.laptop_camera is not None:
                ret, frame = self.laptop_camera.read()
                if ret and frame is not None:
                    return True, frame
                else:
                    print("❌ Ошибка чтения с встроенной камеры")
                    return False, None
            else:
                return self.get_simulation_frame()
        except Exception as e:
            print(f"❌ Ошибка встроенной камеры: {e}")
            return self.get_simulation_frame()

    def get_simulation_frame(self):
        """Создание тестового кадра если камеры недоступны"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "SIMULATION MODE - NO CAMERA", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to exit", (50, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return True, frame

    def switch_to_laptop_camera(self):
        """Переключение на камеру ноутбука"""
        if self.current_camera_type == "DRONE":
            # Закрываем соединение с дроном
            try:
                self.pioneer.disarm()
                print("✅ Дрон отключен")
            except:
                pass
            
            self.drone_connected = False
            self.current_camera_type = "LAPTOP"
            
            # Инициализируем камеру ноутбука если еще не инициализирована
            if self.laptop_camera is None:
                self.laptop_camera = self.init_laptop_camera()
            
            print("🔄 Переключение на камеру ноутбука")

    def cleanup(self):
        """Очистка ресурсов всех камер"""
        # Закрываем соединение с дроном
        if hasattr(self, 'pioneer'):
            try:
                self.pioneer.disarm()
                print("✅ Дрон отключен")
            except Exception as e:
                print(f"⚠️  Ошибка при отключении дрона: {e}")

        # Закрываем камеру ноутбука
        if self.laptop_camera is not None:
            self.laptop_camera.release()
            print("✅ Камера ноутбука закрыта")

        print(f"✅ Ресурсы камер освобождены. Использовалась: {self.current_camera_type} камера")