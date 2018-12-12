import numpy as np
import argparse
import cv2
from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils
import time


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required=True,
                    help='path to input video')
parser.add_argument('-p', '--prototxt', required=True,
                    help="path to 'Caffe' deploy prototxt file")
parser.add_argument('-m', '--model', required=True,
                    help="path to Caffe pretrained model")
parser.add_argument('-c', '--confidence', type=float, default=0.5,
                    help="minimum probability to filter weak detections")
args = vars(parser.parse_args())


# load the serialized model from the disk
print("[INFO] loading model")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream")
vs = FileVideoStream(args["video"]).start()
time.sleep(2.0)

while vs.more():
    frame = vs.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=400)

    (h, w) = frame.shape[:2]
    # create blob
    blob = cv2.dnn.blobFromImage(
            cv2.resize(
                    frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass through nnet to get detections
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(detections.shape[2]):
        # get the confidence level
        confidence = detections[0, 0, i, 2]

        # weed out weak detections
        if confidence > args["confidence"]:

            # compute the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype('int')

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame,
                          (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame,
                        text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 2)

    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & (0xFF)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
