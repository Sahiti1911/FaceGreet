import cv2
import face_recognition
import os
import sqlite3
from datetime import datetime
import pyttsx3

class FaceRecognitionAttendance:
    def __init__(self):
        self.greeted = {}
        self.image_path = 'ImagesAttendance'
        self.class_names = []
        self.encode_list_known = []
        self.greet_timeout = 300  # Greet again after 5 minutes (300 seconds)
        self.db_conn = None

    def connect_to_database(self):
        self.db_conn = sqlite3.connect('attendance.db')
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date_time DATETIME NOT NULL
            )
        ''')
        self.db_conn.commit()

    def load_images_and_encodings(self):
        myList = os.listdir(self.image_path)
        for cl in myList:
            cur_img = cv2.imread(os.path.join(self.image_path, cl))
            name = os.path.splitext(cl)[0]
            self.class_names.append(name)

            # Encode faces
            cur_img_rgb = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(cur_img_rgb)[0]
            self.encode_list_known.append(encode)
            self.greeted[name] = {'last_greet_time': None}

    def mark_attendance(self, name):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor = self.db_conn.cursor()
        cursor.execute("INSERT INTO attendance (name, date_time) VALUES (?, ?)", (name, current_time))
        self.db_conn.commit()

    def greet_person(self, name):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        current_time = datetime.now()
        greeting = ""

        if current_time.hour < 12:
            greeting = 'Good morning, '

        elif 12 <= current_time.hour < 18:
            greeting = 'Good afternoon, '

        else:
            greeting = 'Good evening, '

        for voice in voices:
            engine.setProperty('voice', voice.id)
            text = f"{greeting}{name}"
            engine.say(text)
            engine.runAndWait()

    def check_greet_timeout(self, name):
        current_time = datetime.now()
        last_greet_time = self.greeted.get(name, {}).get('last_greet_time', None)

        if last_greet_time is None:
            return True  # Greet if no previous greet time is recorded

        time_difference = (current_time - last_greet_time).total_seconds()
        return time_difference >= self.greet_timeout

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

            faces_cur_frame = face_recognition.face_locations(img_small_rgb)
            encodes_cur_frame = face_recognition.face_encodings(img_small_rgb, faces_cur_frame)

            recognized_names = []
            for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
                matches = face_recognition.compare_faces(self.encode_list_known, encode_face)
                face_dis = face_recognition.face_distance(self.encode_list_known, encode_face)
                match_index = min(range(len(matches)), key=lambda i: face_dis[i])

                if matches[match_index]:
                    name = self.class_names[match_index].upper()
                    recognized_names.append(name)

                    if self.check_greet_timeout(name):
                        self.mark_attendance(name)
                        self.greet_person(name)
                        if name in self.greeted:
                            self.greeted[name]['last_greet_time'] = datetime.now()

            cv2.imshow('Webcam', img)

            # Check for 'q' key press to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
        self.db_conn.close()

if __name__ == "__main__":
    attendance_system = FaceRecognitionAttendance()
    attendance_system.connect_to_database()
    attendance_system.load_images_and_encodings()
    attendance_system.run()