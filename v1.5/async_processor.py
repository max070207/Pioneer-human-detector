from imports import *

class AsyncProcessor:
    def __init__(self, processing_function: Callable[[np.ndarray], np.ndarray], max_queue_size: int = 2, timeout: float = 2.0) -> None:
        self.processing_function: Callable[[np.ndarray], np.ndarray] = processing_function
        self.max_queue_size: int = max_queue_size
        self.timeout: float = timeout
        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)
        self.output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running: bool = False
        self.last_result: Optional[np.ndarray] = None
        
    def start(self) -> None:
        """Запуск потока обработки"""
        if self.is_running:
            return
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
    def stop(self) -> None:
        """Остановка потока обработки"""
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
    def process_async(self, data: np.ndaaray) -> Optional[np.ndarray]:
        """Асинхронная обработка данных. Возвращает последний результат если новый еще не готов"""
        try:
            self.input_queue.put(data, block=False)
        except queue.Full:
            pass
        try:
            self.last_result = self.output_queue.get(block=False)
        except queue.Empty:
            pass
        return self.last_result
        
    def _processing_loop(self) -> None:
        """Основной цикл обработки в отдельном потоке"""
        while self.is_running:
            try:
                data = self.input_queue.get(timeout=0.1)
                result = self.processing_function(data)
                try:
                    self.output_queue.put(result, block=False)
                except queue.Full:
                    try:
                        self.output_queue.get(block=False)
                        self.output_queue.put(result, block=False)
                    except queue.Empty:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Async processing error: {e}")
                continue