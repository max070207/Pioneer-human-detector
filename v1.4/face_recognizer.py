from imports import *
from config import FACE_RECOGNITION_CONFIG, ASYNC_CONFIG, DATABASE_PATH, FACES_FOLDER

def face_worker(input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue, config: Dict[str, Any]) -> None:
    """Процесс распознавания лиц"""
    face_database: Dict[str, Any] = {}
    database_path: str = config['database_path']
    faces_folder: str = config['faces_folder']
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
    face_search_active: bool = False
    face_found: bool = False
    last_saved_face: Optional[str] = None
    save_message_time: float = 0
    face_save_count: Dict[str, int] = {}
    last_processed_frame: Optional[np.ndarray] = None
    last_result: Optional[Dict[str, Any]] = None
    last_frame_count: int = 0
    while True:
        try:
            latest_task = None
            while True:
                try:
                    task = input_queue.get_nowait()
                    if task is None:
                        if latest_task is not None:
                            input_queue.put(latest_task)
                        break
                    latest_task = task
                except:
                    break
            if latest_task is None:
                time.sleep(0.001)
                continue
            frame, frame_count, human_detected, command = latest_task
            if command == 'reset':
                face_search_active = False
                face_found = False
                last_saved_face = None
                save_message_time = 0
            if not human_detected:
                if face_found or face_search_active:
                    face_search_active = False
                    face_found = False
                    last_saved_face = None
                    save_message_time = 0
                output_queue.put(({
                    'recognized_persons': [],
                    'face_search_active': False,
                    'face_found': False,
                    'last_saved_face': None
                }, frame_count))
                continue
            if human_detected and not face_search_active and not face_found:
                face_search_active = True
            skip_processing = False
            if (last_processed_frame is not None and 
                last_result is not None and
                frame_count - last_frame_count < 3):
                if frame.shape == last_processed_frame.shape:
                    frame_diff = cv2.absdiff(frame, last_processed_frame).mean()
                    if frame_diff < 10:
                        skip_processing = True
            if not skip_processing:
                processing_interval = 1
                recognized_persons_data = []
                if frame_count % processing_interval == 0:
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        model_type = config['model']
                        face_locations = face_recognition.face_locations(rgb_frame, model=model_type)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        last_processed_frame = frame.copy()
                        last_frame_count = frame_count
                        current_found_faces = []
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            person_name = "Unknown"
                            best_match_distance = 1.0
                            for db_name, db_encoding in face_database.items():
                                face_distance = face_recognition.face_distance([db_encoding], face_encoding)[0]
                                if face_distance < config['tolerance'] and face_distance < best_match_distance:
                                    best_match_distance = face_distance
                                    person_name = db_name
                            similarity_percent = (1 - best_match_distance) * 100 if person_name != "Unknown" else 0.0
                            if person_name != "Unknown":
                                current_found_faces.append(person_name)
                                if person_name not in face_save_count:
                                    face_save_count[person_name] = 0
                                
                                if last_saved_face != person_name: #face_save_count[person_name] >= 1:
                                    face_save_count[person_name] += 1
                                    base_name = os.path.splitext(person_name)[0]
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"{base_name}_{face_save_count[person_name]}_{timestamp}.jpg"
                                    if not os.path.exists(faces_folder):
                                        os.makedirs(faces_folder)
                                    cv2.imwrite(os.path.join(faces_folder, filename), frame)
                            recognized_persons_data.append((person_name, (top, right, bottom, left), similarity_percent))
                        if not current_found_faces:
                            last_saved_face = None
                        else:
                            for person_name, location, similarity in recognized_persons_data:
                                if person_name != "Unknown":
                                    last_saved_face = person_name
                                    save_message_time = time.time()
                                    break
                    except Exception as e:
                        recognized_persons_data = []
                final_persons = recognized_persons_data
                last_result = {
                    'recognized_persons': final_persons,
                    'face_search_active': face_search_active,
                    'face_found': len([p for p in final_persons if p[0] != "Unknown"]) > 0,
                    'last_saved_face': last_saved_face,
                    'save_message_time': save_message_time
                }
            output_queue.put((last_result, frame_count))
        except Exception as e:
            continue

class FaceRecognizer:
    def __init__(self, file_manager: Any, log_maker: Any) -> None:
        self.file_manager: Any = file_manager
        self.log_maker: Any = log_maker
        self.logfile_name: str = self.file_manager.get_logfile_name()
        self.face_search_active: bool = False
        self.face_found: bool = False
        self.last_saved_face: Optional[str] = None
        self.save_message_time: float = 0
        self.last_saved_face_similarity: float = 0
        self.input_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
        self.output_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
        self.process: Optional[multiprocessing.Process] = None
        self.latest_result: List[Tuple[str, Tuple[int, int, int, int], float]] = []
        if ASYNC_CONFIG['face_processing']:
            self.start_process()
            
    def start_process(self) -> None:
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

    def process_faces(self, raw_frame: np.ndarray, frame_count: int, is_human_detected: bool) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """Обработка лиц - оптимизированная для отслеживания"""
        if self.process:
            should_process = (
                is_human_detected or 
                self.face_search_active or 
                self.face_found or
                frame_count % 1 == 0
            )
            if should_process:
                try:
                    command = None
                    self.input_queue.put_nowait((raw_frame.copy(), frame_count, is_human_detected, command))
                except queue.Full:
                    try:
                        while self.input_queue.qsize() > 5:
                            self.input_queue.get_nowait()
                        self.input_queue.put_nowait((raw_frame.copy(), frame_count, is_human_detected, command))
                    except:
                        pass
            try:
                latest_data = None
                latest_frame_count = 0
                while not self.output_queue.empty():
                    data, count = self.output_queue.get_nowait()
                    if count >= latest_frame_count:
                        latest_data = data
                        latest_frame_count = count
                if latest_data:
                    self.latest_result = latest_data['recognized_persons']
                    self.face_search_active = latest_data['face_search_active']
                    self.face_found = latest_data['face_found']
                    self.last_saved_face = latest_data['last_saved_face']
                    self.save_message_time = latest_data.get('save_message_time', 0)
                    if self.latest_result and len(self.latest_result) > 0:
                        for person_name, location, similarity in self.latest_result:
                            if person_name != "Unknown":
                                self.last_saved_face_similarity = similarity
                                break
            except queue.Empty:
                pass
        return self.latest_result

    def draw_faces_and_message(self, frame: np.ndarray, recognized_persons: List[Tuple[str, Tuple[int, int, int, int], float]]) -> np.ndarray:
        """Отрисовка лиц и сообщений о всех распознанных лицах"""
        known_faces = []
        for person_name, (top, right, bottom, left), similarity_percent in recognized_persons:
            if self._is_valid_face_coordinates(left, top, right, bottom, frame):
                if person_name != "Unknown":
                    color = (0, 255, 0)
                    known_faces.append((person_name, (top, right, bottom, left), similarity_percent))
                else:
                    color = (0, 0, 255)
                self._draw_face_rectangle(frame, left, top, right, bottom, color)
        if known_faces:
            frame = self._draw_multiple_faces_message_pil(frame, known_faces)
        return frame

    def _is_valid_face_coordinates(self, left: int, top: int, right: int, bottom: int, frame: np.ndarray) -> bool:
        """Проверка валидности координат лица"""
        frame_height, frame_width = frame.shape[:2]
        if (left >= right or top >= bottom or left < 0 or top < 0 or right > frame_width or bottom > frame_height):
            return False
        face_width = right - left
        face_height = bottom - top
        if face_width < 30 or face_height < 30:
            return False
        return True

    def _draw_face_rectangle(self, frame: np.ndarray, left: int, top: int, right: int, bottom: int, color: Tuple[int, int, int]) -> None:
        """Отрисовка прямоугольника вокруг лица"""
        thickness = 2
        distance = (abs(top-bottom))
        lines = [((left, top), (left+distance//5, top)), ((right, top), (right-distance//5, top)),
                ((right, top), (right, top+distance//5)), ((right, bottom), (right, bottom-distance//5)),
                ((right, bottom), (right-distance//5, bottom)), ((left, bottom), (left+distance//5, bottom)),
                ((left, bottom), (left, bottom-distance//5)), ((left, top), (left, top+distance//5))]
        for line in lines:
            cv2.line(frame, line[0], line[1], color, thickness)

    def _draw_multiple_faces_message_pil(self, frame: np.ndarray, known_faces: List[Tuple[str, Tuple[int, int, int, int], float]]) -> np.ndarray:
        """Отрисовка информации о нескольких распознанных лицах через PIL"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img, 'RGBA')
        try:
            font = ImageFont.truetype("verdanab.ttf", 14)
        except:
            font = ImageFont.load_default()
        main_text = f"РАСПОЗНАНО ЛИЦ: {len(known_faces)}"
        try:
            main_bbox = draw.textbbox((0, 0), main_text, font=font)
            main_width = main_bbox[2] - main_bbox[0]
            main_height = main_bbox[3] - main_bbox[1]
            max_name_width = 0
            total_height = main_height
            for i, (person_name, location, similarity) in enumerate(known_faces):
                if not isinstance(person_name, str):
                    person_name = str(person_name)
                display_name = os.path.splitext(person_name)[0]
                face_text = f"{i+1}. {display_name} ({similarity:.1f}%)"
                face_bbox = draw.textbbox((0, 0), face_text, font=font)
                face_width = face_bbox[2] - face_bbox[0]
                face_height = face_bbox[3] - face_bbox[1]
                max_name_width = max(max_name_width, face_width)
                total_height += face_height + 5
            max_width = max(main_width, max_name_width)
            text_height = total_height
        except:
            max_width = 350
            text_height = 90
        frame_height, frame_width = frame.shape[:2]
        padding = 10
        rect_x1 = 0
        rect_y1 = frame_height - 5 - text_height - 10
        rect_x2 = rect_x1 + max_width + 15
        rect_y2 = frame_height
        if rect_y1 < 0:
            rect_y1 = 10
            rect_y2 = rect_y1 + text_height + 2 * padding
        radius = 25
        points = [(rect_x1, rect_y1), (rect_x2 - radius, rect_y1)]
        center_x, center_y = rect_x2 - radius, rect_y1 + radius
        for angle in range(270, 361, 10):
            rad = np.radians(angle)
            px = center_x + radius * np.cos(rad)
            py = center_y + radius * np.sin(rad)
            points.append((px, py))
        points.append((rect_x2, rect_y1 + radius))
        points.append((rect_x2, rect_y2))
        points.append((rect_x1, rect_y2))
        points.append((rect_x1, rect_y1))
        draw.polygon(points, fill=(128, 128, 128, 128))
        text_x = rect_x1 + 5
        text_y = rect_y1 + 5
        draw.text((text_x, text_y), main_text, font=font, fill=(224, 255, 255, 255))
        text_y += main_height + 5
        for i, (person_name, location, similarity) in enumerate(known_faces):
            if not isinstance(person_name, str):
                person_name = str(person_name)
            display_name = os.path.splitext(person_name)[0]
            face_text = f"{i+1}. {display_name} ({similarity:.1f}%)"
            draw.text((text_x, text_y), face_text, font=font, fill=(127, 255, 0, 255))
            try:
                face_bbox = draw.textbbox((0, 0), face_text, font=font)
                face_height = face_bbox[3] - face_bbox[1]
                text_y += face_height + 5
            except:
                text_y += 30
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return result

    def cleanup(self) -> None:
        """Очистка ресурсов"""
        if self.process:
            self.input_queue.put(None)
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()