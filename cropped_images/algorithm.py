from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys
import types

from scipy.stats import wasserstein_distance
from PIL import Image

from abc import ABCMeta, abstractmethod

class Context:
    def __init__(self, strategy):
        self.prepare = strategy.prepare

    # context_interface
    def calculate(self, dataA, dataB):
        return self.prepare(dataA, dataB)

class Strategy(metaclass=ABCMeta):
    @abstractmethod
    # algorithm interface
    def prepare(self, dataA, dataB):
        pass

class EarthMovers(Strategy):
    # algorithm interface
    def prepare(self, dataA, dataB):
        '''
        Measure the Earth Mover's distance between two images
        @args:
            {str} img_a: formatted image
            {str} img_b: formatted image
        @returns:
            TODO
        '''
        #img_a = get_img(path_a, norm_exposure=True)
        #img_b = get_img(path_b, norm_exposure=True)
        hist_a = get_histogram(dataA)
        hist_b = get_histogram(dataB)
        return wasserstein_distance(hist_a, hist_b)

class PercentDifference(Strategy):
    # algorithm interface
    def prepare(self, dataA, dataB):
        '''
        @args:
            {numpy.ndarray} imageA: image that has been formatted
            {numpy.ndarray} imageB: image that has been formatted
        @returns:
            percentage of difference between both images
        '''

        i1 = Image.open(dataA)
        i2 = Image.open(dataB)

        i1 = i1.resize((1000,1000))
        i2 = i2.resize((1000,1000))
        # make sure both images have the same mode = black and white
        i1 = i1.convert("L")    
        i2 = i2.convert("L")

        assert i1.mode == i2.mode, "Different kinds of images."
        assert i1.size == i2.size, "Different sizes."

        pairs = zip(i1.getdata(), i2.getdata())
        if len(i1.getbands()) == 1:
            # for gray-scale jpegs
            dif = sum(abs(p1-p2) for p1,p2 in pairs)
        else:
            dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))

        ncomponents = i1.size[0] * i1.size[1] * 3
        percentage = (dif / 255.0 * 100) / ncomponents

        return percentage

class Ssim(Strategy):
    # algorithm interface
    def prepare(self, dataA, dataB):
        return ssim(dataA, dataB)

class Mse(Strategy):
    # algorithm interface
    def prepare(self, dataA, dataB):
        '''
        @note:
            tests how much error there are between images starting
            from top left to bottom right
        @args:
            {numpy.ndarray} imageA: image that has been formatted
            {numpy.ndarray} imageB: image that has been formatted
        @returns:
            the amount of error between the images
        '''
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((dataA.astype("float") - dataB.astype("float")) ** 2)
        err /= float(dataA.shape[0] * dataA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

def get_histogram(data):
    '''
    @args:
        {str} img: the name of an image
    @returns:
        histogram representation of img

    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    h, w = data.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[data[i, j]] += 1
    return np.array(hist) / (h * w)

