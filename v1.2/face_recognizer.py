from imports import *
import multiprocessing
from config import DATABASE_PATH, FACE_RECOGNITION_CONFIG, ASYNC_CONFIG, FACES_FOLDER

def face_worker(input_queue, output_queue, config):
    """Worker process for face recognition"""
    import face_recognition
    import cv2
    import os
    import time
    
    # Initialize state
    face_database = {}
    database_path = config['database_path']
    faces_folder = config['faces_folder']
    
    # Load database
    if os.path.exists(database_path):
        for filename in os.listdir(database_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(database_path, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        face_database[filename] = encodings[0]
                except:
                    pass

    # State variables
    face_search_active = False
    face_found = False
    last_saved_face = None
    last_saved_face_location = None
    last_saved_face_similarity = 0
    save_message_time = 0
    face_save_count = {}
    
    while True:
        try:
            task = input_queue.get()
            if task is None:
                break
                
            frame, frame_count, human_detected, command = task
            
            # Handle commands
            if command == 'reset':
                face_search_active = False
                face_found = False
                last_saved_face = None
                last_saved_face_location = None
                save_message_time = 0
                
            if not human_detected:
                if face_found or face_search_active:
                    # Reset if human lost
                    face_search_active = False
                    face_found = False
                    last_saved_face = None
                    last_saved_face_location = None
                    save_message_time = 0
                output_queue.put(({
                    'recognized_persons': [],
                    'face_search_active': False,
                    'face_found': False,
                    'last_saved_face': None
                }, frame_count))
                continue

            # Auto-start search if human detected and not found/searching
            if human_detected and not face_search_active and not face_found:
                face_search_active = True

            # Processing logic
            processing_interval = 1 if face_found else 2
            recognized_persons_data = []
            
            if frame_count % processing_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                model_type = config['model']
                face_locations = face_recognition.face_locations(rgb_frame, model=model_type)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                if face_found:
                    # Tracking mode
                    best_match_for_saved = None
                    best_distance_for_saved = 0.7
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        for db_name, db_encoding in face_database.items():
                            if db_name == last_saved_face:
                                face_distance = face_recognition.face_distance([db_encoding], face_encoding)[0]
                                if face_distance < best_distance_for_saved:
                                    best_distance_for_saved = face_distance
                                    best_match_for_saved = (top, right, bottom, left, face_distance)
                                    break
                    if best_match_for_saved:
                        top, right, bottom, left, face_distance = best_match_for_saved
                        last_saved_face_location = (top, right, bottom, left)
                        last_saved_face_similarity = (1 - face_distance) * 100
                
                if not face_found:
                    # Search mode
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        person_name = "Unknown"
                        best_match_distance = 1.0
                        
                        for db_name, db_encoding in face_database.items():
                            face_distance = face_recognition.face_distance([db_encoding], face_encoding)[0]
                            if face_distance < config['tolerance'] and face_distance < best_match_distance:
                                best_match_distance = face_distance
                                person_name = db_name
                        
                        similarity_percent = (1 - best_match_distance) * 100 if person_name != "Unknown" else 0.0
                        
                        if not face_found and person_name != "Unknown":
                            # Found new face - Save it
                            face_found = True
                            last_saved_face = person_name
                            last_saved_face_location = (top, right, bottom, left)
                            last_saved_face_similarity = similarity_percent
                            save_message_time = time.time()
                            face_search_active = False
                            
                            # Save to file
                            if person_name not in face_save_count:
                                face_save_count[person_name] = 0
                            face_save_count[person_name] += 1
                            base_name = os.path.splitext(person_name)[0]
                            filename = f"{base_name}_{face_save_count[person_name]}.jpg"
                            if not os.path.exists(faces_folder):
                                os.makedirs(faces_folder)
                            cv2.imwrite(os.path.join(faces_folder, filename), frame)
                            break
                        
                        recognized_persons_data.append((person_name, (top, right, bottom, left), similarity_percent))

            # Prepare output
            final_persons = []
            if face_found and last_saved_face_location:
                 if time.time() - save_message_time < 5:
                     final_persons.append((last_saved_face, last_saved_face_location, last_saved_face_similarity))
                 else:
                     last_saved_face_location = None
            elif not face_found:
                final_persons = recognized_persons_data

            output_queue.put(({
                'recognized_persons': final_persons,
                'face_search_active': face_search_active,
                'face_found': face_found,
                'last_saved_face': last_saved_face,
                'save_message_time': save_message_time
            }, frame_count))

        except Exception as e:
            # print(f"Face worker error: {e}")
            continue

class FaceRecognizer:
    def __init__(self, file_manager, log_maker) -> None:
        self.file_manager = file_manager
        self.log_maker = log_maker
        self.logfile_name = self.file_manager.get_logfile_name()
        
        self.face_search_active = False
        self.face_found = False
        self.last_saved_face = None
        self.save_message_time = 0
        
        self.input_queue = multiprocessing.Queue(maxsize=1)
        self.output_queue = multiprocessing.Queue(maxsize=1)
        self.process = None
        self.latest_result = []
        
        if ASYNC_CONFIG['face_processing']:
            self.start_process()
            
    def start_process(self):
        config = {
            'database_path': DATABASE_PATH,
            'faces_folder': FACES_FOLDER,
            'model': FACE_RECOGNITION_CONFIG.get('model', 'hog'),
            'tolerance': FACE_RECOGNITION_CONFIG['tolerance']
        }
        self.process = multiprocessing.Process(
            target=face_worker,
            args=(self.input_queue, self.output_queue, config),
            daemon=True
        )
        self.process.start()

    def process_faces(self, raw_frame, frame_count, is_human_detected):
        """Обработка лиц - оптимизированная для отслеживания"""
        if self.process:
            # Send task
            try:
                command = None
                self.input_queue.put_nowait((raw_frame.copy(), frame_count, is_human_detected, command))
            except queue.Full:
                pass
                
            # Get result
            try:
                while not self.output_queue.empty():
                    data, _ = self.output_queue.get_nowait()
                    self.latest_result = data['recognized_persons']
                    self.face_search_active = data['face_search_active']
                    self.face_found = data['face_found']
                    self.last_saved_face = data['last_saved_face']
                    self.save_message_time = data.get('save_message_time', 0)
            except queue.Empty:
                pass
                
        return self.latest_result

    def draw_faces_and_message(self, frame, recognized_persons):
        """Отрисовка лиц и сообщения о сохранении"""
        # Draw recognized faces (search mode)
        if self.face_search_active and not self.face_found:
            for person_name, (top, right, bottom, left), similarity_percent in recognized_persons:
                if self._is_valid_face_coordinates(left, top, right, bottom, frame):
                    color = (255, 255, 0) if person_name != "Unknown" else (0, 0, 255)
                    self._draw_face_rectangle(frame, left, top, right, bottom, color)
        
        # Draw saved face (tracking mode)
        is_tracking_period = self.face_found and (time.time() - self.save_message_time < 5 if self.save_message_time else False)
        if is_tracking_period and recognized_persons:
            for person_name, (top, right, bottom, left), similarity_percent in recognized_persons:
                 if self._is_valid_face_coordinates(left, top, right, bottom, frame):
                    color = (0, 255, 0)
                    self._draw_face_rectangle(frame, left, top, right, bottom, color)
            self._draw_save_message(frame)
            
        return frame

    def _is_valid_face_coordinates(self, left, top, right, bottom, frame) -> bool:
        """Проверка валидности координат лица"""
        frame_height, frame_width = frame.shape[:2]
        if (left >= right or top >= bottom or left < 0 or top < 0 or right > frame_width or bottom > frame_height):
            return False
        face_width = right - left
        face_height = bottom - top
        if face_width < 30 or face_height < 30:
            return False
        return True

    def _draw_face_rectangle(self, frame, left, top, right, bottom, color) -> None:
        """Отрисовка прямоугольника вокруг лица"""
        thickness = 2
        distance = (abs(top-bottom))
        lines = [((left, top), (left+distance//5, top)), ((right, top), (right-distance//5, top)),
                ((right, top), (right, top+distance//5)), ((right, bottom), (right, bottom-distance//5)),
                ((right, bottom), (right-distance//5, bottom)), ((left, bottom), (left+distance//5, bottom)),
                ((left, bottom), (left, bottom-distance//5)), ((left, top), (left, top+distance//5))]
        for line in lines:
            cv2.line(frame, line[0], line[1], color, thickness)

    def _draw_save_message(self, frame) -> None:
        """Отрисовка сообщения о сохранении лица"""
        frame_height = frame.shape[0]
        name = self.last_saved_face if self.last_saved_face else "Unknown"
        if not isinstance(name, str):
            name = str(name)
        message = f"FOUND FACE: {os.path.splitext(name)[0]}"
        cv2.putText(frame, message, (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def cleanup(self) -> None:
        """Очистка ресурсов"""
        if self.process:
            self.input_queue.put(None)
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()