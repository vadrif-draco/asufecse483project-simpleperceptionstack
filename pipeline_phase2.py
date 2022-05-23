# Imports
import os
import cv2
import numpy as np

# Load YOLO weights and configuration, and coco names
yolo_weights = os.path.join("..", "YOLOv3-416", "yolov3.weights")
yolo_cfg = os.path.join("..", "YOLOv3-416", "yolov3.cfg")

# Use configuration and weights to construct neural network
net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
layers = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()  # Output layers are unconnected
output_layers = [layers[index - 1] for index in output_layers_indices]


def detect_car(image_BGR: cv2.Mat):

    HEIGHT, WIDTH = np.shape(image_BGR)[:2]  # Don't need channel

    blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416), crop=False)
    net.setInput(blob)
    net_outputs = net.forward(output_layers)

    bounding_boxes_to_draw = []
    bounding_boxes_confidences = []
    bounding_boxes_nms_indices = []

    for output in net_outputs:

        for cell in output:
            detection = cell[4]
            bounding_box = cell[0:4]
            classes_confidences = cell[5:]

            # winner_class = np.argmax(classes_confidences)  # returns indices of max values in vector
            winner_class = 2
            confidence_of_winner_class = classes_confidences[winner_class]
            # if (detection > 0.55 and confidence_of_winner_class > 0.83):
            if (confidence_of_winner_class > 0.83):
                bounding_boxes_confidences.append(confidence_of_winner_class)

                # The winner class in this case is 2, and if we go to coco names we find that class 2 (i.e. #3) is car
                # The coordinates of the bounding box are given as percentages of the image's resolution, so...
                bounding_box_upsampled = (bounding_box * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])).astype("int")

                # It's time to draw the rectangle... but wait, the coordinates here are box center + box dimensions
                # Need to convert to corner coordinates; namely topleft and bottomright corners
                bounding_box_ready_to_draw = [
                    int(bounding_box_upsampled[0] - bounding_box_upsampled[2] / 2),  # topleft_x
                    int(bounding_box_upsampled[1] - bounding_box_upsampled[3] / 2),  # topleft_y
                    int(bounding_box_upsampled[0] + bounding_box_upsampled[2] / 2),  # bottright_x
                    int(bounding_box_upsampled[1] + bounding_box_upsampled[3] / 2),  # bottright_y
                ]

                bounding_boxes_to_draw.append(bounding_box_ready_to_draw)
                bounding_boxes_nms_indices = cv2.dnn.NMSBoxes(
                    bounding_boxes_to_draw, bounding_boxes_confidences, 0.8, 0.8)

    return bounding_boxes_to_draw, bounding_boxes_confidences, bounding_boxes_nms_indices


def test_pipeline():

    # video_input = cv2.VideoCapture("./assets/project_video.mp4")
    video_input = cv2.VideoCapture("./assets/challenge_video.mp4")
    ret, frame_BGR = video_input.read()

    while(ret):

        bounding_boxes_to_draw, bounding_boxes_confidences, bounding_boxes_nms_indices =\
            detect_car(frame_BGR)

        for i in bounding_boxes_nms_indices:
            box = bounding_boxes_to_draw[i]
            cv2.rectangle(frame_BGR, box[:2], box[2:4], (0, 0, 255), 4)
            cv2.putText(frame_BGR, f'Car @{int(10000*bounding_boxes_confidences[i])/100}%', (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            
        cv2.imshow("Preview", frame_BGR)
        cv2.waitKey(1)

        ret, frame_BGR = video_input.read()


if __name__ == "__main__": test_pipeline()
