import os
from picamera2 import Picamera2
import numpy as np
import cv2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()


count=0

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 300

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
         frame= picam2.capture_array()
         count += 1
         if count % 5 != 0:
            continue
         frame=cv2.flip(frame,-1)
         cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
         cv2.imshow('FRAME', frame)
         if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        frame= picam2.capture_array()
        frame=cv2.flip(frame, 0)
        cv2.imshow('FRAME', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()