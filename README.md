# Face Detection and Labeling with MTCNN
# By: Maryam Mahmoudi
Date: 1 Nov 2022

This Python script uses the MTCNN (Multi-Task Cascaded Convolutional Neural Networks) library to detect faces in images, crop them, and store them as numpy arrays in a binary file using pickle. It also labels the images based on the folder they are located in and stores the labels in a separate binary file.

# Prerequisites
Python 3
TensorFlow 2.x
MTCNN library
OpenCV (cv2)
Scikit-learn
# How to use
Place all the images you want to detect faces on in a folder named "smile_dataset_600".
Inside that folder, create subfolders for each category of images.
Run the script.
The script will create two binary files, "datas" and "labels", containing the data and labels respectively.
# Output
"datas" binary file: Numpy array containing the cropped face images.
"labels" binary file: Numpy array containing the labels for each face image.
Console output: Progress updates and total time taken to process all images.
# Note
This script assumes that there is only one face in each image.
If there is no face detected in an image, it will skip that image.
