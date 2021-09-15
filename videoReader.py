
import time

import cv2
import torch
from Tools.scripts.abitype import classify
from torch import device, half
from torchvision.transforms import ToTensor

from models.experimental import attempt_load
from resize_video import resize_vid
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, check_requirements, check_suffix, colorstr, is_ascii, \
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, \
    save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync


import cv2
from threading import Thread
import sys
import urllib
import json


def inRegion(c1, c2, rois):
    cx = (c1[0] + c2[0]) // 2
    cy = (c1[1] + c2[1]) // 2
    for i in range(len(rois)):
        roi = rois[i]
        if roi[0] < cx < roi[2] and roi[1] < cy < roi[3]:
            return i
    return None


class VideoReader(object):
    """
    This class is for a VideoReader Object that will run on single
    video using rois and will process the frames for occupancy and
    ppe detection
    Attributes:
            streamPort(str or int)	: The path of the videofile or the webcam port
            rois(list(list)) 	: The nested list of rois
            windowName(str)		: Window name to show the output frame  
    """

    def __init__(self, name, streamArg):
        """
        Parameters:
                name(str)       : camera name
                streamArd(dict) : camera data

        """
        super(VideoReader, self).__init__()

        self.window = name
        # cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(self.window, 640, 420)

        self.streamLink = streamArg["streamLink"]
        self.rois = streamArg["rois"]
        self.drawn_roi = self.rois
        self.occupancyTime = [0] * len(self.rois)
        self.roi_kit_label = [None] * len(self.rois)

        self.capture = None
        self.dataset = None
        self.startFeed()
        if self.capture.isOpened():
            self.imgsz = 640
            self.max_det = 1000
            self.processing_device = select_device('cpu')
            self.model = attempt_load(
                'cabin_model.pt', map_location=self.processing_device)
            self.stride = int(self.model.stride.max())
            self.names = self.model.module.names if hasattr(
                self.model, 'module') else self.model.names
            self.imgsz = check_img_size(self.imgsz, s=self.stride)
            self.ascii = is_ascii(self.names)
            self.dt, self.seen = [0.0, 0.0, 0.0], 0

            # For videoFiles only
            self.dataset = LoadImages(self.streamLink, img_size=self.imgsz,
                                      stride=self.stride, auto='.pt')

    def startFeed(self):
        self.capture = cv2.VideoCapture(self.streamLink)
        if not self.capture.isOpened():
            print(f"Capture at {self.streamLink} is not opened")

    def read_frame(self):
        ret, self.frame = self.capture.read()
        assert(ret)

    def show_frame(self):
        im = cv2.resize(self.frame, (640, 420))
        cv2.imshow(self.window, im)

    def run(self):
        while True:
            self.read_frame()
            self.show_frame()
            if cv2.waitKey(1) & 0xff == 27:
                break

    def end(self):
        self.capture.release()

    def process(self):

        prev_frame_time = 0
        new_frame_time = 0

        for path, img, im0s, vid_cap in self.dataset:
            frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
            t1 = time_sync()

            new_frame_time = time.time()
            img = torch.from_numpy(img).to(self.processing_device)
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            t2 = time_sync()
            self.dt[0] += t2 - t1
            pred = self.model(img, augment=False, visualize=False)[0]

            t3 = time_sync()
            self.dt[1] += t3 - t2

            self.roi_kit_label = [None] * len(self.rois)
            # NMS
            pred = non_max_suppression(
                pred, 0.50, 0.45, None, False, max_det=self.max_det)
            self.dt[2] += time_sync() - t3

            for i, det in enumerate(pred):  # per image
                c1 = None
                self.seen += 1
                p, s, im0, frame = path, '', im0s.copy(), getattr(self.dataset, 'frame', 0)

                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                annotator = Annotator(im0, line_width=3, pil=not ascii)
                if len(det):

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])
                                  ), (int(xyxy[2]), int(xyxy[3]))

                        roi_index = inRegion(c1, c2, self.drawn_roi)

                        if roi_index != None:
                            self.occupancyTime[roi_index] += 1/frame_rate
                            self.roi_kit_label[roi_index] = self.names[c]

                        label = f'{self.names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
                print(f'{s}Done. ({t3 - t2:.3f}s)')

                im0 = annotator.result()
                font = cv2.FONT_HERSHEY_SIMPLEX
                for i in range(len(self.drawn_roi)):
                    xmin, ymin, xmax, ymax = self.drawn_roi[i]
                    cv2.putText(im0, f"Roi: {i}", (xmin + 10, ymin - 50), font, 2,
                                (255, 255, 255), 3, cv2.LINE_AA)

                    cv2.rectangle(im0, (xmin, ymin),
                                  (xmax, ymax), (255, 255, 255), 1)

                    cv2.putText(im0, str(round(self.occupancyTime[i], 2)), (xmin + 10, ymin + 50), font, 2,
                                (100, 255, 0), 3, cv2.LINE_AA)
                    try:
                        cv2.putText(im0, self.roi_kit_label[i], (xmin + 10, ymax - 50), font, 2,
                                    (100, 255, 0), 3, cv2.LINE_AA)
                    except:
                        pass

                im0 = cv2.resize(im0, (620, 400))
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps = int(fps)
                fps = str(fps)

                cv2.putText(im0, fps, (7, 70), font, 3,
                            (100, 255, 0), 3, cv2.LINE_AA)

                cv2.imshow(self.window, im0)

                if cv2.waitKey(1) & 0xff == 27:
                    return

            t = tuple(x / self.seen * 1E3 for x in self.dt)


if __name__ == '__main__':

    devices = json.loads(open('camera_ports.json').read())

    vrs = []

    for i, j in devices.items():
        vr = VideoReader(i, j)
        vrs.append(vr)

    threads = []

    for vr in vrs:
        threads.append(Thread(target=vr.process, args=()))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    sys.exit()
