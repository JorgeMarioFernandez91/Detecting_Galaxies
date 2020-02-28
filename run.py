#skimage implements SSIM
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
from scipy.stats import wasserstein_distance
from PIL import Image

# global dictionary to store image_name: formatted_image key:value pairs
img_dict = {}
compared_img_list = []

def get_histogram(img):
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
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / (h * w)

def earth_movers_distance(img_a, img_b):
    '''
    Measure the Earth Mover's distance between two images
    @args:
        {str} path_a: the path to an image file
        {str} path_b: the path to an image file
    @returns:
        TODO
    '''
    #img_a = get_img(path_a, norm_exposure=True)
    #img_b = get_img(path_b, norm_exposure=True)
    hist_a = get_histogram(img_a)
    hist_b = get_histogram(img_b)
    return wasserstein_distance(hist_a, hist_b)

# MSE formula
def mse(imageA, imageB):
    '''
    @note:
        tests how much error there are between images starting
        from top left to bottom right
    @args:
        {numpy.ndarray} imageA: image that has been formatted
        {numpy.ndarray} imageB: image that has been formatted
    @returns:
        histogram representation of img
    '''
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

# take in two images, then calculate MSE and SSIM and return values
def compare_image(imageA, imageB, title, threshold):
    '''
    @note:
        compares luminance, contrast, and structure of both images
    @args:
        {numpy.ndarray} imageA: image that has been formatted
        {numpy.ndarray} imageB: image that has been formatted
        {int} title: a numerical title
    @yields:
        histogram representation of img
    '''

    result1 = []
    result2 = []

    # check if images have already been compared, if so then don't compare otherwise compare
    for img_pair in compared_img_list:
        if img_pair[0] is imageA and img_pair[1] is imageB:
            return
        if img_pair[1] is imageA and img_pair[0] is imageB:
            return

    # add image pair to a list to make sure they do not get compared again
    compared_img_list.append([imageA, imageB])
    compared_img_list.append([imageB, imageA])

    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    temp1 = ''
    temp2 = ''

    for key in img_dict:
        if (img_dict[key] is imageA):
            temp1 = key
        if (img_dict[key] is imageB):
            temp2 = key

    difference = compare_percent_similar(temp1, temp2)

    # show the images if they're above a certain threshold
    if s >= threshold and s < 1.00 and difference < 50:
        # setup the figure
        fig = plt.figure(title)
        plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

        emd = earth_movers_distance(imageA, imageB)

        print("Compare %s to %s: " % (temp1, temp2))
        print("MSE: %.2f" % (m))
        print("SSIM: %.2f" % (s))
        print("Earth Mover's Distance: %.5f" % emd)
        print("Difference (percentage): %.2f" % (difference))
        print('\n')

        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA, cmap = plt.cm.gray)
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB, cmap = plt.cm.gray)
        plt.axis("off")
        plt.show()

def format_images(location, sharpen):
    '''
    @args:
        {str} imageA: the directory of where the images are to be found
        {boolen} imageB: if true then the images will be sharpened, otherwise keep them as is
    @yields:
        updated image dictionary
    '''
    sim_list = []
    galaxy_list = []
    entries = os.listdir(location)
    for entry in entries:
        if entry.endswith(('.jpg', '.png')):
            # load the images
            temp = cv.imread(entry)
            # resizing the images
            temp = cv.resize(temp, (1000, 1000))
            # converting to grayscale
            temp = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)

            if "fof" in entry or "realism" in entry or "subfind" in entry:
                # sharpen the image
                if sharpen is True:
                    kernel = np.array([[-1,-1,-1],
                                       [-1, 9,-1],
                                       [-1,-1,-1]])
                    temp = cv.filter2D(temp, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
                    #cv.imshow('Image Sharpening', sharpened)
                sim_list.append(temp)
            else:
                if "inv" in entry:
                    # inverse colors if needed
                    temp = cv.bitwise_not(temp)
                # save images in list
                galaxy_list.append(temp)

            # storing name of the image and it's converted equal to global dictionary
            img_dict[entry] = temp

def compare_percent_similar(img1, img2):
    '''
    @args:
        {numpy.ndarray} imageA: image that has been formatted
        {numpy.ndarray} imageB: image that has been formatted
    @returns:
        percentage of difference between both images
    '''

    i1 = Image.open(img1)
    i2 = Image.open(img2)

    i1 = i1.resize((150,150))
    i2 = i2.resize((150,150))

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

def main():
    '''
    @yields
        a comparison between all images after they've been formatted to be in greyscale and the same size
    '''

    print('\nLow Threshold\n')

    threshold = 0.3

    format_images('.', False) # path to images

    num = 0
    for key1 in img_dict:
        if(key1.find("sim") is not -1): # if the image we retrieved is a simulated image then we do want to compare it with real images
            img1 = img_dict[key1]
            for key2 in img_dict:
                if(key2.find("sim") is -1): # if the image we retrieved is a real galaxy image then we do want to compare it with simulated images
                    img2 = img_dict[key2]
                    compare_image(img1, img2, num, threshold)
                    num += 1


    print('\nUn-contrasted\n')

    threshold = 0.6

    globals()['img_dict'] = {}
    globals()['compared_img_list'] = []

    format_images('.', False) # path to images

    num = 0
    for key1 in img_dict:
        if(key1.find("sim") is not -1): # if the image we retrieved is a simulated image then we do want to compare it with real images
            img1 = img_dict[key1]
            for key2 in img_dict:
                if(key2.find("sim") is -1): # if the image we retrieved is a real galaxy image then we do want to compare it with simulated images
                    img2 = img_dict[key2]
                    compare_image(img1, img2, num, threshold)
                    num += 1

    print('\nContrasted\n')

    threshold = 0.6

    globals()['img_dict'] = {}
    globals()['compared_img_list'] = []

    format_images('.', True) # path to images

    num = 0
    for key1 in img_dict:
        if(key1.find("sim") is not -1): # if the image we retrieved is a simulated image then we do want to compare it with real images
            img1 = img_dict[key1]
            for key2 in img_dict:
                if(key2.find("sim") is -1): # if the image we retrieved is a real galaxy image then we do want to compare it with simulated images
                    img2 = img_dict[key2]
                    compare_image(img1, img2, num, threshold)
                    num += 1

if __name__ == "__main__":
    # execute only if run as a script
    main()
