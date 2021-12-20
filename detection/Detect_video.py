from .Security import Security
from .WhatsappCall import WhatappCall as wpp
import numpy as np
import cv2
import time
import os

path_file = os.path.dirname(os.path.abspath(__file__))

# Detecting Objects on Video with OpenCV deep learning library
#
# How it works:
# Reading input video --> Loading YOLO v3 Network -->
# --> Reading frames in the loop --> Getting blob from the frame -->
# --> Implementing Forward Pass --> Getting Bounding Boxes -->
# --> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
# --> Writing processed frames
#
# Result:
# New video file with Detected Objects, Bounding Boxes and Labels

#Video = '1_3_trabalhadores3.mp4'

class Detection():

    

    def __init__(self, video_path):
        """ Constructior responsible for creating the Detection instance """

        self.video_path = str(video_path)
        
        

        # Preparing variables for spatial dimensions of the frames
        self.h = None
        self.w = None 
        
        # Preparing variable for saving images
        self.counter = 0

        

        # Setting probability minimum (all objects that are detected with les prob will be discarted)
        self.probability_minimum = 0.5

    def read_input_video(self):
        """ Defining 'VideoCapture' object and reading video from a file"""

        self.video = cv2.VideoCapture(self.video_path)
    
    def load_yolo_networks(self):

        # Loading COCO class labels from file

        with open(path_file+'\yolo-coco-data\coco.names') as f:
            # Getting labels reading every line
            # and putting them into the list
            self.labels_coco = [line.strip() for line in f]

        with open(path_file+'\Custom_dataset\hat.names') as f:
            # Getting labels reading every line
            # and putting them into the list
            self.labels_hat = [line.strip() for line in f]

        with open(path_file+'\Custom_dataset\mask.names') as f:
            # Getting labels reading every line
            # and putting them into the list
            self.labels_mask = [line.strip() for line in f]

        self.labels = [self.labels_coco[0:1], self.labels_hat, self.labels_mask]

        labels_coco= self.labels_coco
        labels_hat = self.labels_hat
        labels_mask = self.labels_mask

        # Loading trained YOLO Objects Detector
        # with the help of 'dnn' library from OpenCV

        self.network_coco = cv2.dnn.readNetFromDarknet(path_file+'\yolo-coco-data\yolov3.cfg',
                                                path_file+'\yolo-coco-data\yolov3.weights')

        self.network_hat = cv2.dnn.readNetFromDarknet(
            path_file+'\Custom_dataset\hat.cfg',
            path_file+'\Custom_dataset\hat.weights')

        self.network_mask = cv2.dnn.readNetFromDarknet(
            path_file+'\Custom_dataset\mask.cfg',
            path_file+'\Custom_dataset\mask.weights')

        # Getting list with names of all layers from YOLO network
        layers_names_all_coco = self.network_coco.getLayerNames()
        layers_names_all_hat = self.network_hat.getLayerNames()
        layers_names_all_mask = self.network_mask.getLayerNames()

        self.layers_names_output_coco = [layers_names_all_coco[i[0] - 1] for i in self.network_coco.getUnconnectedOutLayers()]

        self.layers_names_output_hat = \
            [layers_names_all_hat[i[0] - 1] for i in self.network_hat.getUnconnectedOutLayers()]

        self.layers_names_output_mask = \
            [layers_names_all_mask[i[0] - 1] for i in self.network_mask.getUnconnectedOutLayers()]

        # Setting threshold for filtering weak bounding boxes
        # with non-maximum suppression
        self.threshold = 0.3

        # Generating colours for representing every detected object
        # with function randint(low, high=None, size=None, dtype='l')
        self.colours_coco = np.random.randint(0, 255, size=(len(labels_coco), 3), dtype='uint8')
        self.colours_hat = np.random.randint(0, 255, size=(len(labels_hat), 3), dtype='uint8')
        self.colours_mask = np.random.randint(0, 255, size=(len(labels_mask), 3), dtype='uint8')

    def read_frames(self):

        self.delta_start = 0
        self.delta_detected = 0

        # Defining variable for counting frames
        # At the end we will show total amount of processed frames
        f = 0

        # Defining variable for counting total time
        # At the end we will show time spent for processing all frames
        t = 0

        self.writer = None # Preparing variable for writer that we will use to write processed frames

        # Defining loop for catching frames
        while True:
            # Capturing frame-by-frame
            ret, frame = self.video.read()

            # If the frame was not retrieved >>> at the end of the video, BREAK LOOP
            if not ret:
                break

            # Getting spatial dimensions of the frame
            if self.w is None or self.h is None:
                # Slicing from tuple only first two elements
                self.h, self.w = frame.shape[:2]

            # Getting blob from current frame
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                        swapRB=True, crop=False)

            """
            Implementing Forward pass
            """

            # Implementing forward pass with our blob and only through output layers
            # Calculating at the same time, needed time for forward pass
            self.network_coco.setInput(blob)  # setting blob as input to the network
            self.network_hat.setInput(blob)  # setting blob as input to the network
            self.network_mask.setInput(blob)  # setting blob as input to the network
            start = time.time()
            output_from_network_coco = self.network_coco.forward(self.layers_names_output_coco)
            output_from_network_hat = self.network_hat.forward(self.layers_names_output_hat)
            output_from_network_mask = self.network_mask.forward(self.layers_names_output_mask)
            end = time.time()

            # Increasing counters for frames and total time
            f += 1
            t += end - start

            # Showing spent time for single current frame
            print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))


            """
            Getting bounding boxes
            """

            # Preparing lists for detected bounding boxes, obtained confidences, class's number and models number
            # resets every frame
            self.bounding_boxes = []
            self.confidences = []
            self.class_numbers = []
            self.model = []  # 0 - Coco, 1 - Hat, 2 - mask

            self.get_bounding_boxes(output_from_network_coco,0)
            self.get_bounding_boxes(output_from_network_hat,1)
            self.get_bounding_boxes(output_from_network_mask,2)

            """
            Non-maximum suppression

            Implementing non-maximum suppression of given bounding boxes
            with this technique we exclude some of bounding boxes if their
            corresponding confidences are low or there is another
            bounding box for this region with higher confidence

            """

            self.results = cv2.dnn.NMSBoxes(self.bounding_boxes, self.confidences,
                                    self.probability_minimum, self.threshold)


            """
            Drawing bounding boxes and labels
            """

            # Checking if there is at least one detected object
            # after non-maximum suppression
            if len(self.results) > 0:
                # Going through indexes of results
                for i in self.results.flatten():
                    # Getting current bounding box coordinates,
                    # its width and height
                    x_min, y_min = self.bounding_boxes[i][0], self.bounding_boxes[i][1]
                    box_width, box_height = self.bounding_boxes[i][2], self.bounding_boxes[i][3]

                    # Preparing colour for current bounding box
                    # and converting from numpy array to list
                    if self.model[i] == 0:
                        colour_box_current = self.colours_coco[self.class_numbers[i]].tolist()
                    elif self.model[i] == 1:
                        colour_box_current = self.colours_hat[self.class_numbers[i]].tolist()
                    elif self.model[i] == 2:
                        colour_box_current = self.colours_mask[self.class_numbers[i]].tolist()

                    # Drawing bounding box on the original current frame
                    cv2.rectangle(frame, (x_min, y_min),
                                (x_min + box_width, y_min + box_height),
                                colour_box_current, 2)

                    # Preparing text with label and confidence for current bounding box

                    if self.model[i] == 0:
                        text_box_current = '{}: {:.4f}'.format(self.labels_coco[int(self.class_numbers[i])],
                                                            self.confidences[i])
                    elif self.model[i] == 1:
                        text_box_current = '{}: {:.4f}'.format(self.labels_hat[int(self.class_numbers[i])],
                                                            self.confidences[i])
                    elif self.model[i] == 2:
                        text_box_current = '{}: {:.4f}'.format(self.labels_mask[int(self.class_numbers[i])],
                                                            self.confidences[i])

                    # Putting text with label and confidence on the original image
                    cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

            """
            Writing processed frame into the file
            """

            # Initializing writer
            # we do it only once from the very beginning
            # when we get spatial dimensions of the frames
            if self.writer is None:
                # Constructing code of the codec
                # to be used in the function VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                # Writing current processed frame into the video file
                self.writer = cv2.VideoWriter(self.video_path[:-4] + '_VF.mp4', fourcc, 30,
                                        (frame.shape[1], frame.shape[0]), True)

            # Write processed current frame to the file
            self.writer.write(frame)

            #Checking if the given frame is secure
            self.check_security(frame)

            print('-' * 20)

        # Printing final results
        print()
        print('Total number of frames', f)
        print('Total amount of time {:.5f} seconds'.format(t))
        print('FPS:', round((f / t), 1))

        # Releasing video reader and writer
        self.video.release()
        self.writer.release()

    def get_bounding_boxes(self, output_from_network, model_number):
        for result in output_from_network:
                # Going through all detections from current output layer
                for detected_objects in result:
                    # Getting 80 classes' probabilities for current detected object
                    scores = detected_objects[5:]
                    # Getting index of the class with the maximum value of probability
                    class_current = np.argmax(scores)
                    # Getting value of probability for defined class
                    confidence_current = scores[class_current]

                    # Eliminating weak predictions with minimum probability
                    if confidence_current > self.probability_minimum:
                        # Scaling bounding box coordinates to the initial frame size
                        # YOLO data format keeps coordinates for center of bounding box
                        # and its current width and height
                        # That is why we can just multiply them elementwise
                        # to the width and height
                        # of the original frame and in this way get coordinates for center
                        # of bounding box, its width and height for original frame
                        box_current = detected_objects[0:4] * np.array([self.w, self.h, self.w, self.h])

                        # Now, from YOLO data format, we can get top left corner coordinates
                        # that are x_min and y_min
                        x_center, y_center, box_width, box_height = box_current
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))

                        # Adding results into prepared lists
                        self.bounding_boxes.append([x_min, y_min,
                                            int(box_width), int(box_height)])
                        self.confidences.append(float(confidence_current))
                        self.class_numbers.append(class_current)
                        self.model.append(model_number)

    def check_security(self, frame):
        if not Security.isFrameSecure(self.model, self.labels, self.bounding_boxes, self.class_numbers, self.results):
            if self.delta_start == 0:
                self.delta_start = time.time()
            else:
                delta_end = time.time()
                delta = delta_end - self.delta_start
                print(delta)
                if delta > 10:
                    self.delta_start = time.time()
                    if self.delta_detected == 0 or delta_end - self.delta_detected > 120:
                        self.delta_detected = time.time()
                        file = 'images/screenshot_notsafe/{0}NOTSAFE{1}.jpg'.format("video_", str(self.counter))
                        cv2.imwrite(file, frame)
                        wpp.SendWhatsApp('+5511949314360', file)
                        self.counter += 1
        else:
            self.delta_start = time.time()

    def detect(self):
        self.read_input_video()
        self.load_yolo_networks()
        self.read_frames()
