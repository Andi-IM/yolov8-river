import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO('best.pt')

image = Image.open('full.png')

results = model(image)

speed = 5 # in px/s
max_height = 1077.2202 # taken from last maximum record

print(results)

plot = results[0].plot()
boxes = results[0].boxes
box = boxes[0]
data = box.xywh
numpy_data = data.detach().to('cpu').numpy()
print("numpy data = ", numpy_data)
detected_height = numpy_data[0][3]
detected_width = numpy_data[0][2]
print("water_speed = ", speed, " px/s (cuma dipatok doang)")
print("current_width = ", detected_width, "px")
print("current_height = ", detected_height, "px")
estimated_time = (max_height - detected_height) / speed 
print("estimated time = ", "{:10.2f}".format(estimated_time), " s")
# cv2.imshow("result", plot)

cv2.waitKey(0)
cv2.destroyAllWindows()
    

