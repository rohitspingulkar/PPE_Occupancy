import queue
import cv2
import threading

class Capture:
    def __init__(self,camAddr):
        #frame capture thread
        self.camAddr = camAddr
        self.cap = cv2.VideoCapture(self.camAddr)
        self.frameQue = queue.Queue()
        self.framCapFlag = 1

    def startCaptureThread(self):
        self.cap = cv2.VideoCapture(self.camAddr)
        self.camThread = threading.Thread(target=self._reader)
        self.camThread.daemon = True
        self.camThread.start()
        
    def stopCaptureThread(self):
        self.framCapFlag = 0
        self.camThread.join()
        self.release()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("break from queue")
                self.cap = cv2.VideoCapture(self.camAddr)
                continue
            if not self.frameQue.empty():
                try:
                    self.frameQue.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.frameQue.put(frame)
            if(not self.framCapFlag):
                break
    
    def read(self):
        return self.frameQue.get()

    def release(self):
        self.cap.release()