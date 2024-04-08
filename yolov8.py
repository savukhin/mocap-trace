import cv2
import torch
import numpy as np
from PIL import Image
import ultralytics
from ultralytics import YOLO
ultralytics.checks()

# Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov8', 'custom', path='./yolov8x.pt', source='github')
model = YOLO("bestv8.pt") 
# model = YOLO("yolov8x.pt") 
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', source='github')

# Set video source (webcam or video file)
cap = cv2.VideoCapture(0)

# Define the classes you want to detect
classes = ['Drone']

# Initialize the rectangle coordinates
rectangle_coords = [(50, 50), (250, 50), (250, 250), (50, 250)]
rectangle_drag = False
drag_corner = -1

# Function to handle mouse events
def mouse_event(event, x, y, flags, param):
    global rectangle_coords, rectangle_drag, drag_corner

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the mouse click is near any of the rectangle corners
        for i, corner in enumerate(rectangle_coords):
            if abs(corner[0] - x) <= 10 and abs(corner[1] - y) <= 10:
                rectangle_drag = True
                drag_corner = i
                break

    elif event == cv2.EVENT_LBUTTONUP:
        rectangle_drag = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle_drag:
            rectangle_coords[drag_corner] = (x, y)

# Create a window and set the mouse event callback function
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_event)
img_counter = 0
names = model.names

while True:
    # Read frame from video source
    ret, frame = cap.read()
    if not ret:
        print("cannot read frame")
        continue

    # Convert the frame to a format that YOLOv5 can process
    img = [frame]
    # img = Image.fromarray(frame[...,::-1])

    # Run inference on the frame
    # results = model(img, size=480)
    img_path = "path/opencv_frame_{}.png".format(img_counter)
    results = model(img)
    # results = model.predict(img_path, frame)
    img_counter += 1
    # print(results)

    for result in results:
        # print(result)
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs
        print(len(boxes))
        # for box in boxes:
        #     print(box)
        #     print("tolist", box.xyxy.tolist())
        # print("boxes", boxes)
        # print("masks", masks)
        # print("probs", probs)
        # r = result.tolist()
        # print(r)


        # Process the results and draw bounding boxes on the frame
        for box in boxes:
            # print("tolist")
            boxlist = box.xyxy.tolist()[0]
            print(boxlist)
            x1, y1, x2, y2 = boxlist
            # x1, y1, x2, y2 = boxlist[0], boxlist[1], boxlist[2], boxlist[3]
            conf = box.conf[0]
            # conf = box.conff
            cls = box.cls.tolist()
            print("cls", cls)
            droneIndex = -1
            for i, c in enumerate(cls):
                print("names[int(c)]", names[int(c)])
                if names[int(c)] == "drone":
                    droneIndex = i
                    continue
            if droneIndex == -1:
                continue

            print("conf", conf, conf > 0.1)
            print("droneIndex", droneIndex)
            # conf = conf[droneIndex]
            if conf > 0.1:
            # if conf > 0.5 and classes[int(cls)] in classes:
            # if conf > 0.5 and classes[int(cls)] in classes:
                # Draw the bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                print("rect created")

                # Display the confidence score above the box
                text_conf = "{:.2f}%".format(conf * 100)
                cv2.putText(frame, text_conf, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Display the bounding box coordinates below the box
                text_coords = "({}, {})".format(int((x1 + x2) / 2), int(y2))
                cv2.putText(frame, text_coords, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Check if the drone intersects with or is inside the rectangle
                if rectangle_coords[0] != rectangle_coords[1]:
                    if any(rectangle_coords[0][0] <= x <= rectangle_coords[2][0] and rectangle_coords[0][1] <= y <= rectangle_coords[2][1] for x, y in
                        [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]) or \
                            any(rectangle_coords[0][0] <= x <= rectangle_coords[2][0] and rectangle_coords[0][1] <= y <= rectangle_coords[2][1] for x in range(int(x1), int(x2))
                                for y in range(int(y1), int(y2))):
                        # Display a warning message
                        cv2.putText(frame, "Warning: Drone Detected Under Restricted Area!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Draw the rectangle
    for i in range(4):
        cv2.circle(frame, rectangle_coords[i], 5, (0, 255, 0), -1)
        cv2.line(frame, rectangle_coords[i], rectangle_coords[(i+1)%4], (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video source and close the window
cap.release()
cv2.destroyAllWindows()
