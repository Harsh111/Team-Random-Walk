# Team-Random-Walk
Stop sign detection

How to run:

Run open.py to iterate over all files in darknet/data/Test and run the stop sign detection on each of them.
The coordinates of the stop-sign are written in objects.txt

test.py is then run to read the coordinates, do the required manipulations and write required csv file.

How it works:

This stop-sign detection model uses Darket.
Darknet is an open source neural network framework written in C and CUDA.
It is fast, easy to install, and supports CPU and GPU computation.

We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.

A single convolutional network simultaneously predicts multiple bounding boxes and class probabilities for those boxes.
Our model trains on full images and directly optimizes detection performance.

Our system divides the input image into an S × S grid.
If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.
Each grid cell predicts B bounding boxes and confidence scores for those boxes.
These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts.

Formally we define confidence as Pr(Object) ∗ IOU(truth,pred). 
If no object exists in that cell, the confidence scores should be zero.
Otherwise we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth.

We implement this model as a convolutional neural network and evaluate it on the PASCAL VOC detection dataset.
The initial convolutional layers of the network extract features from the image while the fully connected layers predict the output probabilities and coordinates.

The model has 24 convolutional layers followed by 2 fully connected layers, we use 1 × 1 reduction layers followed by 3 × 3 convolutional layers.

