import os

dirname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dirname)
import cv2
from ultralytics import YOLO, settings

settings.update({'mlflow': True})
settings.reset()

import mlflow
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO('yolo11m.pt').to(device)

from time import perf_counter
import easydict
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

deepsort = None


def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)


init_tracker()

import numpy as np

cap = cv2.VideoCapture('streams/store.mp4')
tracker_time = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, classes=[0])

    for r in results:
        if len(r.boxes) > 0:
            bbox_xywh = r.boxes.xywh.cpu().numpy().astype('float64')
            confs = r.boxes.conf.cpu().numpy().astype('float64')
            clss = r.boxes.cls.cpu().numpy().astype('float64')

            outputs = deepsort.update(bbox_xywh, confs, clss, frame)

            if len(outputs) > 0:
                for output in outputs:
                    x1, y1, x2, y2, track_id = map(int, output[:5])

                    # time
                    if track_id not in tracker_time:
                        tracker_time[track_id] = [perf_counter(), perf_counter()]
                    tracker_time[track_id][1] = perf_counter()

                    time_in_store = tracker_time[track_id][1] - tracker_time[track_id][0]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"TrackId:{track_id} Time:{time_in_store:.2f}s",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



