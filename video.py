import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO('best.pt')

video_path = 'sample.mp4'
capture = cv2.VideoCapture(video_path)

# Loop through the video frames
while capture.isOpened():
    # read a frame from the video
    success, frame = capture.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # visualize the results on the frame
        annotated_frame = results[0].plot()
    
        # Display the annotated frame

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close display window
capture.release()
cv2.destroyAllWindows()