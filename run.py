#skimage implements SSIM
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import os
import time

from scipy.stats import wasserstein_distance
from PIL import Image



# global dictionary to store image_name: formatted_image key:value pairs
img_dict = {}
compared_img_list = []

total_match = 0
total_different = 0
total_comparisons = 0
total_wrong = 0

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
        {str} img_a: formatted image
        {str} img_b: formatted image
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
        the amount of error between the images
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
def compare_image(imageA, imageB, title, thresholds, show_image, system_test):
    '''
    @note:
        compares luminance, contrast, and structure of both images
    @args:
        {numpy.ndarray} imageA: image that has been formatted
        {numpy.ndarray} imageB: image that has been formatted
        {int} title: a numerical title
    @yields:
        values determining how similar images are: 
        MSE, SSIM, Earth Mover's Distance, Percent different
    '''

    global total_match;
    global total_different;
    global total_comparisons;
    global total_wrong;

    match = False
    same_galaxy = False

    temp1 = ''
    temp2 = ''

    for key in img_dict:
        if (img_dict[key] is imageA):
            temp1 = key
        if (img_dict[key] is imageB):
            temp2 = key

    # check if images have already been compared, if so then don't compare otherwise compare
    for img_pair in compared_img_list:
        if img_pair[0] is imageA and img_pair[1] is imageB:
            return
        if img_pair[1] is imageA and img_pair[0] is imageB:
            return

    # add image pair to a list to make sure they do not get compared again
    compared_img_list.append([imageA, imageB])
    compared_img_list.append([imageB, imageA])

    if (system_test == True):

        # compute the mean squared error and structural similarity
        # index for the images
        m = mse(imageA, imageB)
        s = ssim(imageA, imageB)

        difference = compare_percent_similar(temp1, temp2)
        emd = earth_movers_distance(imageA, imageB)

        # print("Compare %s to %s: " % (temp1, temp2))
        # print("MSE: %.2f" % (m))
        # print("SSIM: %.2f" % (s))
        # print("Earth Mover's Distance: %.5f" % emd)
        # print("Difference (percentage): %.2f\n" % (difference))
 

        if temp1.find("m31") and temp2.find("m31") or temp1.find("m33") and temp2.find("m33") or temp1.find("m81") and temp2.find("m81"):
            same_galaxy = True


        #thresholds = [mse, ssim, earthmovers, percentdiff]
        # show the images if they're above a certain threshold
        if m < thresholds[0] and s >= thresholds[1] and s < 1.00 and emd < thresholds[2] and difference <= thresholds[3]:

            match = True
            # setup the figure
            fig = plt.figure(title)
            plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

            print("Compare %s to %s: " % (temp1, temp2))
            print("MSE: %.2f" % (m))
            print("SSIM: %.2f" % (s))
            print("Earth Mover's Distance: %.5f" % emd)
            print("Difference (percentage): %.2f" % (difference))
            print('\n')
            
            if (show_image == True):
                # show first image
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(imageA, cmap = plt.cm.gray)
                plt.axis("off")

                # show the second image
                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(imageB, cmap = plt.cm.gray)
                plt.axis("off")
                plt.show()
            
        # tallying up if we got matches, differences, wrong answers, and totals of our system
        if match == True and same_galaxy == True:
            total_match += 1
        elif match == False and same_galaxy == True:
            total_wrong += 1
        else:
            total_different += 1

        total_comparisons += 1

    else:   #if we're testing humans
        temp1 = ''
        temp2 = ''

        for key in img_dict:
            if (img_dict[key] is imageA):
                temp1 = key
                #print(temp1)
            if (img_dict[key] is imageB):
                temp2 = key
                #print(temp2)
       
        #print("left image name: " + temp1)

        fig = plt.figure(title)
        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA, cmap = plt.cm.gray)
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB, cmap = plt.cm.gray)
        plt.axis("off")

        plt.draw()
        plt.pause(0.001)
        ans = input("Press [1] or [2] then [enter] to continue.\n")
        plt.close()


        # check if input is correct
        if ans is '1':  # if user enters 'galaxies match'

            if temp1.find("m31") and temp2.find("m31"):
                total_match += 1
            
            elif temp1.find("m33") and temp2.find("m33"):
                total_match += 1

            elif temp1.find("m81") and temp2.find("m81"):
                total_match += 1
            else:
                total_wrong += 1

        elif ans is '2': # if user enters 'galaxies do not match'

            if temp1.find("m31") and temp2.find("m33") or temp1.find("m31") and temp2.find("m81"):
                total_different += 1

            elif temp1.find("m33") and temp2.find("m31") or temp1.find("m33") and temp2.find("m81"):
                total_different += 1

            elif temp1.find("m81") and temp2.find("m31") or temp1.find("m81") and temp2.find("m33"):
                total_different += 1
            else:
                total_wrong += 1
        else:
            total_wrong += 1
        
        total_comparisons += 1

        # plt.show(block=False)
        
        # time.sleep(5)
        # plt.close(fig)
        # os.system("TASKKILL /F /IM matplotlib")

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
           
            # print(entry)
            # plt.hist(temp.ravel(),256,[0,256]); plt.show()
            # plt.show()
            # exit()

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


def test_system(sim_name, galaxy_name, thresholds, sharpen, show_image):
    
    format_images('.', sharpen) # path to images

    #print(img_dict)

    num = 0
    for key1 in img_dict:
        if(key1.find("sim") is not -1 and key1.find(sim_name) is not -1): # if the image we retrieved is a simulated image then we want to compare it with real images
            img1 = img_dict[key1]
            for key2 in img_dict:
                if(key2.find("sim") is -1 and key2.lower().find(galaxy_name) is not -1): # if the image we retrieved is a real galaxy image then we do want to compare it with simulated images
                    img2 = img_dict[key2]
                    #print(key1 + " " + key2)
                    #print(show_image)
                    compare_image(img1, img2, num, thresholds, show_image, system_test=True)
                    num += 1

def test_human(sim_name, thresholds, sharpen, show_image):
    format_images('.', sharpen) # path to images
    global total_comparisons;
    num = 0
    for key1 in img_dict:
        if(key1.find("realism") is not -1 and key1.find(sim_name) is not -1): # if the image we retrieved is a simulated image then we want to compare it with real images
            img1 = img_dict[key1]
            for key2 in img_dict:
                if(key2.find("sim") is -1): # if the image we retrieved is a real galaxy image then we do want to compare it with simulated images
                    img2 = img_dict[key2]
                    #print(key1 + " " + key2)
                    #print(show_image)
                    compare_image(img1, img2, num, thresholds, show_image, system_test=False)
                    
                    num += 1

def main():
    '''
    @yields
        a comparison between all images after they've been formatted to be in greyscale and the same size
    '''

    # print('\nTesting Human\n')
    
    # test_human(sim_name="m31", threshold=0.2, sharpen=True, show_image=True)
    # test_human(sim_name="m33", threshold=0.2, sharpen=True, show_image=True)
    # test_human(sim_name="m81", threshold=0.2, sharpen=True, show_image=True)

    # print("Total Matches: " + str(total_match))
    # print("Total Different: " + str(total_different))
    # print("Total Wrong: " + str(total_wrong))
    # print("Total Comparisons: " + str(total_comparisons))
    

    # exit()

    global total_match;
    global total_different;
    global total_comparisons;
    global total_wrong;

    #mse ssim earthmovers percent difference
    # values = [4000, 0.1, 0.004, 30]

    # compare a specific galaxy to its simulations
    # print('\nM31\n')
    # globals()['img_dict'] = {}
    # globals()['compared_img_list'] = []
    # test_system(sim_name="m31_realism", galaxy_name="m31", thresholds=values, sharpen=True, show_image=False)
    # #test_system(sim_name="m31", galaxy_name="m31", thresholds=values, sharpen=True, show_image=False)  

    # print("Total Matches: " + str(total_match))
    # print("Total Different: " + str(total_different))
    # print("Total Wrong: " + str(total_wrong))
    # print("Total Comparisons: " + str(total_comparisons))

    # total_match = 0;
    # total_different = 0;
    # total_comparisons = 0;
    # total_wrong = 0;

    # values = [4000, 0.1, 0.004, 30]

    # print('\nM33\n')
    # globals()['img_dict'] = {}
    # globals()['compared_img_list'] = []
    # #test_system(sim_name="m33", galaxy_name="m33", thresholds=values, sharpen=True, show_image=False) 
    # test_system(sim_name="m33_realism", galaxy_name="m33", thresholds=values, sharpen=True, show_image=False)

    # print("Total Matches: " + str(total_match))
    # print("Total Different: " + str(total_different))
    # print("Total Wrong: " + str(total_wrong))
    # print("Total Comparisons: " + str(total_comparisons))

    # exit()

    # total_match = 0;
    # total_different = 0;
    # total_comparisons = 0;
    # total_wrong = 0;

    # values = [4000, 0.1, 0.004, 30]

    # print('\nM81\n')
    # globals()['img_dict'] = {}
    # globals()['compared_img_list'] = []
    # #test_system(sim_name="m81", galaxy_name="m81" , thresholds=values, sharpen=True, show_image=False) 
    # test_system(sim_name="m81_realism", galaxy_name="m81" , thresholds=values, sharpen=True, show_image=False) 

    # print("Total Matches: " + str(total_match))
    # print("Total Different: " + str(total_different))
    # print("Total Wrong: " + str(total_wrong))
    # print("Total Comparisons: " + str(total_comparisons))

    # exit()

    #reading image



    total_match = 0;
    total_different = 0;
    total_comparisons = 0;
    total_wrong = 0;

    values = [4000, 0.1, 0.004, 30]

    # test that non related images are not related
    print('\nSIM M31 vs GALAXY M81\n')
    print('\nContrasted Other Comparison\n')
    globals()['img_dict'] = {}
    globals()['compared_img_list'] = []
    #test_system(sim_name="m31", galaxy_name="m81" , thresholds=values, sharpen=True, show_image=False)
    test_system(sim_name="m31_realism", galaxy_name="m81" , thresholds=values, sharpen=True, show_image=False)  

    print("Total Matches: " + str(total_match))
    print("Total Different: " + str(total_different))
    print("Total Wrong: " + str(total_wrong))
    print("Total Comparisons: " + str(total_comparisons))

    exit()

    print('\nSIM M33 vs GALAXY M31\n')
    globals()['img_dict'] = {}
    globals()['compared_img_list'] = []
    test_system(sim_name="m33", galaxy_name="m31" ,threshold=0.2, sharpen=True, show_image=False) 


    print('\nSIM M81 vs GALAXY M33\n')
    globals()['img_dict'] = {}
    globals()['compared_img_list'] = []
    test_system(sim_name="m81", galaxy_name="m33" , thresholds=values, sharpen=True, show_image=False) 


if __name__ == "__main__":
    # execute only if run as a script
    main()
