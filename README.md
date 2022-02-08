# Detecting Safety Equipment in Construction Sites
This project was created with the goal of detecting safety equipment use on construction sites.

![alt text](images\screenshot_notsafe\video_NOTSAFE0.jpg)

Data show that most of the work accidents happen in the civil construction sector. Every 15 seconds a worker dies due to an accident or illness at work. This statistic reinforces the need to create mechanisms to improve safety in construction sites and, therefore, reduce the number of accidents. 

The most efficient way to achieve this reduction is the correct use of PPE, which stands for Personal Safety Equipment. For this, we can count on technology with the use of artificial intelligence.

In this project I used the [YOLO](https://pjreddie.com/darknet/yolo/)  machine learning algorithm for object detection. [YOLO](https://pjreddie.com/darknet/yolo/) (You Only Look Once), is an efficient object recognition algorithm.

![alt text](images\detection.png)

For this project I trained the algorithm to recognize helmets and seat belts. In addition, the algorithm was trained to identify the lack of use of masks, which are currently required in the workplace under brazilian law due to the covid-19 pandemic. 

After the recognition of objects, the screening of the items viewed by the camera or video images is done by a Python algorithm. When any item indicated by the model is missing, the algorithm will have the ability to take a printscreen of the image or frame at the time and send a message to the cell phone of the person responsible for the safety in a construction site.

![alt text](images\whatsapp.png)

## Feel free to use this code and expand the studies.


