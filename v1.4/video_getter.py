from imports import *

class VideoGetter:
    def __init__(self, src: int = 0):
        self.stream: cv2.VideoCapture = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            self.started: bool = False
            return
        self.started: bool = False
        self.read_lock: threading.Lock = threading.Lock()
        self.grabbed: bool
        self.frame: Optional[np.ndarray] = self.stream.read()
        self.stopped: bool = False
        
    def start(self) -> Any:
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self) -> None:
        while self.started:
            if self.stopped:
                break
            grabbed, frame = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            if not grabbed:
                self.stop()
                
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.read_lock:
            if not self.grabbed:
                return False, None
            return True, self.frame.copy()

    def stop(self) -> None:
        self.started = False
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.stream.release()

    def isOpened(self) -> bool:
        return self.stream.isOpened()