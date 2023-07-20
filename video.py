import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO('best.pt')

video_path = 'sample.mp4'
capture = cv2.VideoCapture(video_path)

speed = 5 # in px/s
max_height = 1077.2202 # taken from last maximum record
# max_width = 1559.206
# max_area = max_width * max_height
# Loop through the video frames
while capture.isOpened():
    # read a frame from the video
    success, frame = capture.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # visualize the results on the frame
        annotated_frame = results[0].plot()
        boxes = results[0].boxes
        if (len(boxes) == 0):
            continue
        else:
            box = boxes[0]
            metrics = box.xywh
            numpy_data = metrics.detach().to('cpu').numpy()
            detected_height = numpy_data[0][3]
            # detected_width = numpy_data[0][2]
            # detected_area = detected_width * detected_height
            estimated_time = (max_height - detected_height) / speed 

            if (estimated_time >= 0):
                time_text = "Estimated time = {:.1f}s".format(estimated_time)
            else:
                time_text = "FULL!!"

            annotated_frame = cv2.putText(
                annotated_frame, 
                f"Water speed = {speed} px/s (note: this is only constanta)",
                (0,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,0,0), 2, cv2.LINE_AA)    
            annotated_frame = cv2.putText(
                annotated_frame, 
                time_text,
                (0,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,0,0), 2, cv2.LINE_AA)
            
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