from collections import deque

import cv2
import numpy as np
from thop.vision.calc_func import calculate_parameters

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("model/best_weights.pt", verbose=False)


# Open the video file or real time webcam
input_video_capture_parameter = 2 # Realtime webcam capture
# input_video_capture_parameter = "./assets/videos/60fps.mp4"
cap = cv2.VideoCapture(input_video_capture_parameter)

tracking_history = {}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot(labels=False)


        # Update tracking history and draw lines
        for track in results[0].boxes:
            track_id = int(track.id[0]) if track.id[0] is not None else 0
            if track_id is not None:
                x_center = int((track.xyxy[0][0] + track.xyxy[0][2]) / 2)
                y_center = int((track.xyxy[0][1] + track.xyxy[0][3]) / 2)
                if track_id not in tracking_history:
                    tracking_history[track_id] = deque(maxlen=500)
                tracking_history[track_id].append((x_center, y_center))
                if len(tracking_history[track_id]) > 1:
                    cv2.polylines(
                        annotated_frame,
                        [np.array(tracking_history[track_id], dtype=np.int32)],
                        isClosed=False,
                        color=(0, 255, 0),
                        thickness=2,
                    )

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

