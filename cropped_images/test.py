import unittest
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys
import os
import types
from scipy.stats import wasserstein_distance
from PIL import Image
from abc import ABCMeta, abstractmethod
from algorithm import *
from run import *

entries = os.listdir('.')

image1 = entries[7]
image2 = entries[21]

# load the images
image1 = cv.imread(image1)
# resizing the images
image1 = cv.resize(image1, (1000, 1000))
# converting to grayscale
image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

# load the images
image2 = cv.imread(image2)
# resizing the images
image2 = cv.resize(image2, (1000, 1000))
# converting to grayscale
image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

imageA = image1
imageB = image1
imageC = image2

class TestRunAndAlgorithmScriptMethods(unittest.TestCase):

    def test_PercentDifference(self):
        imageA = entries[1]
        imageB = entries[1]
        imageC = entries[3]

        difference = Context(PercentDifference())
        difference_equal = difference.calculate(imageA, imageB)
        difference_not_equal = difference.calculate(imageA, imageC)
        
        self.assertEqual(difference_equal, 0)
        self.assertNotEqual(difference_not_equal, 0)

    def test_EartMovers(self):
        emd = Context(EarthMovers())
        emd_equal = emd.calculate(imageA, imageB)
        emd_not_equal = emd.calculate(imageA, imageC)

        self.assertEqual(emd_equal, 0)
        self.assertNotEqual(emd_not_equal, 0)

    def test_Ssim(self):
        ssim = Context(Ssim())
        ssim_equal = ssim.calculate(imageA, imageB)
        ssim_not_equal = ssim.calculate(imageA, imageC)

        self.assertEqual(ssim_equal, 1.0)
        self.assertNotEqual(ssim_not_equal, 1.0)

    def test_Mse(self):
        mse = Context(Mse())
        mse_equal = mse.calculate(imageA, imageB)
        mse_not_equal = mse.calculate(imageA, imageC)

        self.assertEqual(mse_equal, 0)
        self.assertNotEqual(mse_not_equal, 0)
    
    def test_test_system_match(self):

        values = [4000, 0.1, 0.004, 30]
        location = "./test_image_folder/"
        #location = "./"
        results = test_system("test01_sim_m31", "M31(2)", values, True, False, location)

        total_matches = results[0][0]
        total_different = results[0][1]
        total_comparisons = results[0][2]
        total_wrong = results[0][3]

        self.assertEqual(total_matches, 1)
        self.assertEqual(total_different, 0)
        self.assertEqual(total_comparisons, 1)
        self.assertEqual(total_wrong, 0)

    def test_test_system_mismatch(self):
        
        values = [4000, 0.1, 0.004, 30]
        location = "./test_image_folder/"
        #location = "./"
        results = test_system("test01_sim_m31", "M81", values, True, False, location)

        total_matches = results[0][0]
        total_different = results[0][1]
        total_comparisons = results[0][2]
        total_wrong = results[0][3]

        self.assertEqual(total_matches, 1)
        self.assertEqual(total_different, 1)
        self.assertEqual(total_comparisons, 2)
        self.assertEqual(total_wrong, 0)

if __name__ == '__main__':
    unittest.main()