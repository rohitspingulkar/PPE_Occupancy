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


# Necessary arguments
def inRegion(c1, c2, roi):
    cx = (c1[0] + c2[0]) // 2
    cy = (c1[1] + c2[1]) // 2
    if roi[0] < cx < roi[2] and roi[1] < cy < roi[3]:
        return True
    return False


def getRoi(videoPath):
    cap = cv2.VideoCapture(videoPath)
    ret, frame = cap.read()
    xmin, ymin, w, h = cv2.selectROI("Display", frame, False)
    drawn_roi = [xmin, ymin, xmin + w, ymin + h]
    cap.release()
    cv2.destroyAllWindows()

    return drawn_roi


def main():
    print("Programm Starting")

    videoPath = './videos/workerlm2.mp4'
    cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Display", 1024, 640)

    drawn_roi = getRoi(videoPath)
    xmin, ymin, xmax, ymax = drawn_roi
    print(drawn_roi)
    return
    occupancyTime = 0

    prev_frame_time = 0
    new_frame_time = 0
    imgsz = 640
    augment = False
    max_det = 1000
    device = select_device('cpu')
    model = attempt_load(
        'cabin_model.pt', map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)

    dt, seen = [0.0, 0.0, 0.0], 0

    dataset = LoadImages(videoPath, img_size=imgsz,
                         stride=stride, auto='.pt')
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    for path, img, im0s, vid_cap in dataset:
        frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
        t1 = time_sync()

        new_frame_time = time.time()
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        t2 = time_sync()
        dt[0] += t2 - t1
        pred = model(img, augment=augment, visualize=False)[0]

        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, 0.50, 0.45, None, False, max_det=max_det)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):  # per image
            c1 = None
            seen += 1
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

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
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])
                              ), (int(xyxy[2]), int(xyxy[3]))
                    if inRegion(c1, c2, drawn_roi):
                        occupancyTime += 1/frame_rate
                        roi_kit_label = names[c]
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            im0 = annotator.result()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(im0, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
            cv2.putText(im0, str(round(occupancyTime, 2)), (xmin + 10, ymin + 50), font, 2,
                        (100, 255, 0), 3, cv2.LINE_AA)
            try:
                cv2.putText(im0, roi_kit_label, (xmin + 10, ymax - 50), font, 2,
                            (100, 255, 0), 3, cv2.LINE_AA)
            except:
                pass

            im0 = cv2.resize(im0, (1280, 720))
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)

            cv2.putText(im0, fps, (7, 70), font, 3,
                        (100, 255, 0), 3, cv2.LINE_AA)

            # cv2.imshow(str(p), im0)
            cv2.imshow("Display", im0)
            cv2.waitKey(1)
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image


if __name__ == "__main__":
    main()
