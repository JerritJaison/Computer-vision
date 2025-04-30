import cv2
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from util.centroid_tracker import CentroidTracker
import os

# results = model.predict('input_video/sample.mp4',save=True)
# print(results[0])
# for box in results[0].boxes:
#     print(box)

# Load YOLOv5s model
model= torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=True)

# Initialize CentroidTracker
ct=CentroidTracker()
crowd_threshold = 10  # Trigger alert if crowd exceeds this

# Store positions for heatmap
positions =[]

video=cv2.VideoCapture('input_video/sample.mp4')

while video.isOpened():
    ret,frame =video.read()
    if not ret:
        break
    results=model(frame)
    detections=results.pandas().xyxy[0]
    people =detections[detections['name'] =='person']

    rects =[]
    for index,row in people.iterrows():
        x1,y1,x2,y2 =int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        rects.append((x1,y1,x2,y2))
    objects=ct.update(rects)

    # draw boxes and IDs
    for objectID,centroid in objects.items():
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0]-10,centroid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]),4,(0,255,0),-1)
        positions.append((centroid[0], centroid[1]))  # Store for heatmap

    # get people count
    people_count =len(objects)
    cv2.putText(frame, f"People Count: {people_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # alert
    if people_count >crowd_threshold:
        cv2.putText(frame, "ALERT: CROWD TOO DENSE!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Crowd Monitor",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# heatmap
positions=np.array(positions)
heatmap_img=np.zeros((1080,1920))  # Adjust according to your video resolution
for x,y in positions:
    if y < 1080 and x <1920:
        heatmap_img[y][x] +=1

plt.figure(figsize=(10,6))
sns.heatmap(heatmap_img,cmap='hot',cbar=False)
plt.title("Crowd Movement Heatmap")
plt.axis('off')
plt.savefig("heatmap.png")
plt.close()
print("Heatmap saved as 'heatmap.png'")