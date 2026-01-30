from imports import *
import threading
import queue

class VideoGetter:
    """
    Класс для асинхронного захвата видеопотока.
    Позволяет основному циклу не блокироваться на ожидании кадра.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            self.started = False
            return
            
        self.started = False
        self.read_lock = threading.Lock()
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        
    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if self.stopped:
                break
                
            grabbed, frame = self.stream.read()
            
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                
            if not grabbed:
                self.stop()
                
    def read(self):
        with self.read_lock:
            if not self.grabbed:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.stream.release()

    def isOpened(self):
        return self.stream.isOpened()
