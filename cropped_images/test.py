'''
@note:
    This script tests various methods from the mail run.py script 
    to ensure proper output and image comparison
'''

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
# select a simulation and a non related galaxy
image1 = entries[7]
image2 = entries[21]

# load the images, resize and covert to grayscale
image1 = cv.imread(image1)
image1 = cv.resize(image1, (1000, 1000))
image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

image2 = cv.imread(image2)
image2 = cv.resize(image2, (1000, 1000))
image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

imageA = image1
imageB = image1
imageC = image2

class TestRunAndAlgorithmScriptMethods(unittest.TestCase):
    '''
    @note: 
        class which runs unit tests on all methods that start with 'test'
    @yields:
        the results of the tests
    '''

    def test_PercentDifference(self):
        '''
        @note: 
            tests the Percent Difference algorithm
        @yields:
            the percent difference between images, the smaller the percent
            the more similar the images are to each other with 0 being
            a result of idential images. 
            
            Here we assert two things:
            1. if identical images return a value of 0
            2. if different images return a value not equal 0
        '''

        imageA = entries[1]
        imageB = entries[1]
        imageC = entries[3]

        difference = Context(PercentDifference())
        difference_equal = difference.calculate(imageA, imageB)
        difference_not_equal = difference.calculate(imageA, imageC)
        
        self.assertEqual(difference_equal, 0)
        self.assertNotEqual(difference_not_equal, 0)

    def test_EartMovers(self):
        '''
        @note: 
            tests the Earth Mover's algorithm
        @yields:
            a value which represents the amount of work need to do
            to transform one image into the other, the smaller the
            returned value the more similar the images are  
            
            Here we assert two things:
            1. if identical images return a value of 0
            2. if different images return a value not equal 0
        '''

        emd = Context(EarthMovers())
        emd_equal = emd.calculate(imageA, imageB)
        emd_not_equal = emd.calculate(imageA, imageC)

        self.assertEqual(emd_equal, 0)
        self.assertNotEqual(emd_not_equal, 0)

    def test_Ssim(self):
        '''
        @note: 
            tests the Earth Mover's algorithm
        @yields:
            a value which represents the amount of work need to do
            to transform one image into the other, the smaller the
            returned value the more similar the images are  
            
            Here we assert two things:
            1. if identical images return a value of 0
            2. if different images return a value not equal 0
        '''

        ssim = Context(Ssim())
        ssim_equal = ssim.calculate(imageA, imageB)
        ssim_not_equal = ssim.calculate(imageA, imageC)

        self.assertEqual(ssim_equal, 1.0)
        self.assertNotEqual(ssim_not_equal, 1.0)

    def test_Mse(self):
        '''
        @note: 
            tests the Mean Squared Error algorithm
        @yields:
            a value which represents the amount of difference present
            between two images. The higher the return the higher the
            difference between images where 0 indicates identical 
            images
            
            Here we assert two things:
            1. if identical images return a value of 0
            2. if different images return a value not equal 0
        '''

        mse = Context(Mse())
        mse_equal = mse.calculate(imageA, imageB)
        mse_not_equal = mse.calculate(imageA, imageC)

        self.assertEqual(mse_equal, 0)
        self.assertNotEqual(mse_not_equal, 0)
    
    def test_test_system_match(self):
        '''
        @note: 
            tests the method which tests various images and applies
            the various algorithms described above. More specifically,
            we are seeing if the system returns a match when it should
        @yields:
            a tally of total matches, total differences, total comparisons,
            and total wrong choices done by the system
            
            Here we assert four things if there is a match then:
            1. total matches should equal 1
            2. total differences should equal 0
            3. total comparisons should equal 1
            4. total wrong should equal 0
        '''

        values = [4000, 0.1, 0.004, 30]
        location = "./test_image_folder/"
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
        '''
        @note: 
            tests the method which tests various images and applies
            the various algorithms described above. More specifically,
            we are seeing if the system returns a mismatch when it should
        @yields:
            a tally of total matches, total differences, total comparisons,
            and total wrong choices done by the system

            important to note that the system resumes where the previous
            method left off and as such the base values for match, difference,
            comparisons, and wrong are 1,0,1,0 not 0,0,0,0
            
            Here we assert four things if there is a mismatch then:
            1. total matches should equal 1
            2. total differences should equal 1
            3. total comparisons should equal 2
            4. total wrong should equal 0
        '''
        values = [4000, 0.1, 0.004, 30]
        location = "./test_image_folder/"
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