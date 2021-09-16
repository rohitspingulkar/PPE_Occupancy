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

prev_frame_time = 0
new_frame_time = 0


# Necessary arguments
half = False
classify = False
imgsz = 640
augment = False
conf_thres = 0.50,  # confidence threshold
iou_thres = 0.45,
agnostic_nms = False,  # class-agnostic NMS
max_det = 1000
hide_labels = False,  # hide labels
hide_conf = False,
classes = None
device = select_device('cpu')
model = attempt_load('cabin_model.pt', map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names

if half:
    model.half()  # to FP16
if classify:  # second-stage classifier
    modelc = load_classifier(name='resnet50', n=2)  # initialize
    modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)[
                           'model']).to(device).eval()
imgsz = check_img_size(imgsz, s=stride)  # check image size
ascii = is_ascii(names)

dt, seen = [0.0, 0.0, 0.0], 0
webcam = True
if webcam:
    source = "2"
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto='.pt')
    bs = len(dataset)  # batch_size
else:
    dataset = LoadImages('videos/new-3_Trim.mp4', img_size=imgsz,
                         stride=stride, auto='.pt')

bs = 1  # batch_size
vid_path, vid_writer = [None] * bs, [None] * bs
for path, img, im0s, vid_cap in dataset:
    t1 = time_sync()
    new_frame_time = time.time()
    img = torch.from_numpy(img).to(device)
    # print(type(img))
    img = img.half() if half else img.float()  # uint8 to fp16/32

    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1
    pred = model(img, augment=augment, visualize=False)[0]
    t3 = time_sync()
    dt[1] += t3 - t2
    # NMS
    pred = non_max_suppression(pred, 0.50, 0.45, None, False, max_det=max_det)
    # print(pred)
    dt[2] += time_sync() - t3

    for i, det in enumerate(pred):  # per image
        # print(i)
        # print(det)
        seen += 1
        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(
                                ), dataset.count
        else:
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(im0, line_width=3, pil=not ascii)
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
        print(f'{s}Done. ({t3 - t2:.3f}s)')
        im0 = annotator.result()
        im0 = cv2.resize(im0, (1280, 720))
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im0, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
