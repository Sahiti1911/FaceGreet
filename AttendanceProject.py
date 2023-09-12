import cv2
import numpy as np
import face_recognition
import os
import pyttsx3
from gtts import gTTS
from playsound import playsound
from datetime import datetime


# from PIL import ImageGrab
greeted = {}
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    name = os.path.splitext(cl)[0]
    classNames.append(name)
    greeted[name] = False
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            
            if name not in greeted:
                greeted[name] = False
            
            if not greeted[name]:
                markAttendance(name)
                print(name)
                greeted[name] = True  # Set the flag to True after greeting

        else:
            name = 'Unknown'
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            print(name)
            markAttendance(name)
            if name in greeted:
                greeted[name] = False  # Reset the flag for unknown faces


        currentTime = datetime.now()
        currentTime.hour
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if currentTime.hour < 12:
            print('\nGood morning')
            # changing voice
            for voice in voices:
                engine.setProperty('voice', voices[0].id)
                # convert this text to speech
                text = 'Good Morning'
                engine.say(text+name)
            # play the speech
            engine.runAndWait()


        elif 12 <= currentTime.hour < 18:
            print('\nGood afternoon')
            # initialize Text-to-speech engine
            for voice in voices:
                engine.setProperty('voice', voices[0].id)
                # convert this text to speech
                text = 'Good Afternoon'
                engine.say(text+name)
            # play the speech
            engine.runAndWait()


        else:
            print('\nGood evening')
            # initialize Text-to-speech engine
            for voice in voices:
                engine.setProperty('voice', voices[0].id)
                # convert this text to speech
                text = 'Good Evening'
                engine.say(text+name)
            # play the speech
            engine.runAndWait()

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)