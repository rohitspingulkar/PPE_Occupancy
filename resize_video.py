import cv2
import numpy as np


def resize_vid(path):
    cap = cv2.VideoCapture(path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('Input Videos/output.mp4', fourcc, 30, (1280, 720))

    while True:
        ret, frame = cap.read()
        if ret:
            b = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            out.write(b)
        else:
            break
    print('Done Video Resizing')
    cap.release()
    out.release()
    cv2.destroyAllWindows()
