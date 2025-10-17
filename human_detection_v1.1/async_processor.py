# async_processor.py
from imports import *

class AsyncProcessor:
    def __init__(self, processing_function, max_queue_size=2, timeout=2.0):
        self.processing_function = processing_function
        self.max_queue_size = max_queue_size
        self.timeout = timeout
        
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=1)
        self.processing_thread = None
        self.is_running = False
        self.last_result = None
        
    def start(self):
        """Запуск потока обработки"""
        if self.is_running:
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
    def stop(self):
        """Остановка потока обработки"""
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
    def process_async(self, data):
        """
        Асинхронная обработка данных
        Возвращает последний результат если новый еще не готов
        """
        try:
            # Пытаемся добавить в очередь (не блокируя основной поток)
            self.input_queue.put(data, block=False)
        except queue.Full:
            # Очередь полна - используем последний результат
            pass
            
        # Пытаемся получить новый результат
        try:
            self.last_result = self.output_queue.get(block=False)
        except queue.Empty:
            # Нового результата нет - используем старый
            pass
            
        return self.last_result
        
    def _processing_loop(self):
        """Основной цикл обработки в отдельном потоке"""
        while self.is_running:
            try:
                # Ждем данные с таймаутом
                data = self.input_queue.get(timeout=0.1)
                
                # Выполняем обработку
                result = self.processing_function(data)
                
                # Сохраняем результат
                try:
                    self.output_queue.put(result, block=False)
                except queue.Full:
                    # Очередь полна - заменяем старый результат
                    try:
                        self.output_queue.get(block=False)
                        self.output_queue.put(result, block=False)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                # Нет данных для обработки - продолжаем цикл
                continue
            except Exception as e:
                print(f"Async processing error: {e}")
                continue