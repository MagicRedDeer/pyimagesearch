import numpy as np
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True,
                    help="path to an input image")
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

# load the input image and construct an input blob for the image by resizing to
# a fixed 300 X 300 pixels and then normalizing it
image = cv2.imread(args["image"])
h, w = image.shape[:2]
blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob to the network and obtain the predictions and the detections
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
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
