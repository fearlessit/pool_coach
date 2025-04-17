import cv2
from ultralytics import YOLO

model = YOLO("./model/best_weights.pt", task="detect", verbose=True)
video_capture = cv2.VideoCapture(2)

while True:
    ret, frame = video_capture.read()
    results = model.predict(source=frame, verbose=True, show=True)
    for result in results:
        # result.show()
        print(f"ball: {result}")
        print("-------------------")








