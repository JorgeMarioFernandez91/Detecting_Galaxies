import cv2 
import matplotlib.pyplot as plt

#reading image
img1 = cv2.imread('sim_m31_realism_01.png')  
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#keypoints
sift = cv2.SIFT()