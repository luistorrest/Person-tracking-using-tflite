import tensorflow as tf
import cv2
import numpy as np
from picamera2 import Picamera2
from collections import OrderedDict
from scipy.spatial import 
import fun_gps 

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        print(f"Person {self.nextObjectID} entered the scene")
        self.nextObjectID += 1

    def deregister(self, objectID):
        print(f"Person {objectID} left the scene")
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = distance.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)

            usedRows = set()
            usedCols = set()
            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])) - usedRows
            unusedCols = set(range(D.shape[1])) - usedCols

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects


def run(model: str, width: int, height: int, num_threads: int, enable_edgetpu: bool) -> None:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()

    interpreter = tf.lite.Interpreter(model_path=model, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_height, model_width = input_details[0]['shape'][1:3]
    
    PERSON_CLASS_ID = 0
    min_score = 0.5
    tracker = CentroidTracker()

    while True:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        rgb_image_resized = cv2.resize(frame, (model_width, model_height))
        
        if input_details[0]['dtype'] == np.uint8:
            input_data = np.expand_dims(rgb_image_resized, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(rgb_image_resized, axis=0).astype(np.float32) / 255.0        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        rects = []
        for i in range(len(scores)):
            if scores[i] > min_score and int(classes[i]) == PERSON_CLASS_ID:
                ymin, xmin, ymax, xmax = boxes[i]
                h, w, _ = frame.shape
                startX,startY,endX,endY = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
                rects.append((startX,startY,endX,endY))
                
                cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)

        objects = tracker.update(rects)
        for objectID, centroid in objects.items():
            cv2.putText(frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        cv2.imshow('Object Detector', frame)
        if cv2.waitKey(1) == 27:
            break

    picam2.stop()
    cv2.destroyAllWindows()
