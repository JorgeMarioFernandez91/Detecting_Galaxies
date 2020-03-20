import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import sys
import time
from scipy.stats import wasserstein_distance
from PIL import Image
from algorithm import Context, Strategy, Ssim, Mse, EarthMovers, PercentDifference


# global data structures and variables to reference throughout script
img_dict = {}
compared_img_list = []
total_match = 0
total_different = 0
total_comparisons = 0
total_wrong = 0

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

    # reference global variables
    global total_match;
    global total_different;
    global total_comparisons;
    global total_wrong;
    # flags to test if answers are correct/incorrect
    match = False
    same_galaxy = False

    temp1 = ''
    temp2 = ''
    #search image dict for the name of the images to reference them later
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

    #if we are testing the system 
    if (system_test == True):

        # strategy pattern for concealing algorithms
        mse = Context(Mse())
        ssim = Context(Ssim())
        emd = Context(EarthMovers())
        pd = Context(PercentDifference())

        m = mse.calculate(imageA, imageB)
        s = ssim.calculate(imageA, imageB)
        emd = emd.calculate(imageA, imageB)
        difference = pd.calculate(temp1, temp2)
 
        temp1 = temp1.lower()
        temp2 = temp2.lower()

        if temp1.find("m31") > -1 and temp2.find("m31") > -1: 
            same_galaxy = True
        if temp1.find("m33") > -1 and temp2.find("m33") > -1: 
            same_galaxy = True
        if temp1.find("m81") > -1 and temp2.find("m81") > -1:
            same_galaxy = True

        # show the images if they're above a certain threshold
        if m < thresholds[0] and s >= thresholds[1] and s < 1.00 and emd < thresholds[2] and difference <= thresholds[3]:

            match = True
        
            # if the true then show image to the user
            if (show_image == True):
                # setup the figure
                fig = plt.figure(title)
                plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
                # show first image
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(imageA, cmap = plt.cm.gray)
                plt.axis("off")

                # show the second image
                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(imageB, cmap = plt.cm.gray)
                plt.axis("off")
                plt.show()

        log_results(match, same_galaxy)

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

        temp1 = temp1.lower()
        temp2 = temp2.lower()
   
        # check if input is correct
        if ans is '1':  # if user enters 'galaxies match'

            if temp1.find("m31") > -1 and temp2.find("m31") > -1:
                total_match += 1
            
            elif temp1.find("m33") > -1 and temp2.find("m33") > -1:
                total_match += 1

            elif temp1.find("m81") > -1 and temp2.find("m81") > -1:
                total_match += 1
            else:
                total_wrong += 1

        elif ans is '2': # if user enters 'galaxies do not match'

            if temp1.find("m31") > -1 and temp2.find("m31") == -1:
                total_different += 1

            elif temp1.find("m33") > -1 and temp2.find("m33") == -1:
                total_different += 1

            elif temp1.find("m81") > -1 and temp2.find("m81") == -1:
                total_different += 1
            else:
                total_wrong += 1
        
        else: # if user enters something else 
            total_wrong += 1
        
        total_comparisons += 1

        print("Total Matches: " + str(total_match))
        print("Total Different: " + str(total_different))
        print("Total Wrong: " + str(total_wrong))
        print("Total Comparisons: " + str(total_comparisons))  

    return [total_match, total_different, total_comparisons, total_wrong]

def log_results(match, same_galaxy):

    global total_match;
    global total_different;
    global total_comparisons;
    global total_wrong;

    # tallying up if we got matches, differences, wrong answers, and totals of our system
    if match == True and same_galaxy == True:
        total_match += 1
    elif match == False and same_galaxy == True:
        total_wrong += 1
    else:
        total_different += 1

    total_comparisons += 1


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
            #print("!!!"+entry)
            # load the images
            temp = cv.imread(location+entry)
            #print("!!!!"+str(temp))
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
                sim_list.append(temp)
            else:
                if "inv" in entry:
                    # inverse colors if needed
                    temp = cv.bitwise_not(temp)
                # save images in list
                galaxy_list.append(temp)

            # storing name of the image and it's converted equal to global dictionary
            img_dict[entry] = temp
            

def test_system(sim_name, galaxy_name, thresholds, sharpen, show_image, location):
    
    format_images(location, sharpen) # path to images
    num = 0
    results_array = []
    for key1 in img_dict:
        if(key1.find("sim") is not -1 and key1.find(sim_name) is not -1): # if the image we retrieved is a simulated image then we want to compare it with real images  
            img1 = img_dict[key1]
            for key2 in img_dict:
                if (key2.lower().find(galaxy_name.lower()) is not -1):           
                    img2 = img_dict[key2]
                    results_array.append( compare_image(img1, img2, num, thresholds, show_image, system_test=True) )
                    num += 1
    return results_array

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
                    compare_image(img1, img2, num, thresholds, show_image, system_test=False)
                    num += 1

def main():
    '''
    @yields
        a comparison between all images after they've been formatted to be in greyscale and the same size
    '''
    global total_match;
    global total_different;
    global total_comparisons;
    global total_wrong;

    if sys.argv[1] == 'human-test': # test human 

        values = [4000, 0.1, 0.004, 30]

        print('\nTesting Human\n')
        
        test_human(sim_name="m31", thresholds=values, sharpen=True, show_image=True)
        test_human(sim_name="m33", thresholds=values, sharpen=True, show_image=True)
        test_human(sim_name="m81", thresholds=values, sharpen=True, show_image=True)

        print("Total Matches: " + str(total_match))
        print("Total Different: " + str(total_different))
        print("Total Right: " + str(total_match + total_different))
        print("Total Wrong: " + str(total_wrong))
        print("Total Comparisons: " + str(total_comparisons))  

    elif sys.argv[1] == 'system-test': # test system
        
        # mse ssim earthmovers percent difference
        values = [4000, 0.1, 0.004, 30]
        # compare a specific galaxy to its simulations
        globals()['img_dict'] = {}
        globals()['compared_img_list'] = []
        test_system(sim_name="m31_realism", galaxy_name="m31", thresholds=values, sharpen=True, show_image=False, location='./')

        print("=====================================================")
        print("Comparing simM31 to realM31")
        print("Total Matches: " + str(total_match))
        print("Total Different: " + str(total_different))
        print("Total Right: " + str(total_match + total_different))
        print("Total Wrong: " + str(total_wrong))
        print("Total Comparisons: " + str(total_comparisons))  
        print("Accuracy: " + str( format((total_match + total_different)/total_comparisons, '.2f') ))  
        print("=====================================================")

        total_match = 0;
        total_different = 0;
        total_comparisons = 0;
        total_wrong = 0;

        values = [4000, 0.1, 0.004, 30]
        globals()['img_dict'] = {}
        globals()['compared_img_list'] = []
        test_system(sim_name="m33_realism", galaxy_name="m33", thresholds=values, sharpen=True, show_image=False, location='./')

        print("=====================================================")
        print("Comparing simM33 to realM33")
        print("Total Matches: " + str(total_match))
        print("Total Different: " + str(total_different))
        print("Total Right: " + str(total_match + total_different))
        print("Total Wrong: " + str(total_wrong))
        print("Total Comparisons: " + str(total_comparisons))    
        print("Accuracy: " + str( format((total_match + total_different)/total_comparisons, '.2f') ))  
        print("=====================================================")  

        total_match = 0;
        total_different = 0;
        total_comparisons = 0;
        total_wrong = 0;

        values = [4000, 0.1, 0.004, 30]
        globals()['img_dict'] = {}
        globals()['compared_img_list'] = []
        test_system(sim_name="m81_realism", galaxy_name="m81" , thresholds=values, sharpen=True, show_image=False, location='./') 

        print("=====================================================")
        print("Comparing simM81 to realM81")
        print("Total Matches: " + str(total_match))
        print("Total Different: " + str(total_different))
        print("Total Right: " + str(total_match + total_different))
        print("Total Wrong: " + str(total_wrong))
        print("Total Comparisons: " + str(total_comparisons))    
        print("Accuracy: " + str( format((total_match + total_different)/total_comparisons, '.2f') ))  
        print("=====================================================") 

        total_match = 0;
        total_different = 0;
        total_comparisons = 0;
        total_wrong = 0;

        values = [4000, 0.1, 0.004, 30]
        globals()['img_dict'] = {}
        globals()['compared_img_list'] = []
        test_system(sim_name="m31_realism", galaxy_name="m81" , thresholds=values, sharpen=True, show_image=False, location='./')  

        print("=====================================================")
        print("Comparing simM31 to realM81")
        print("Total Matches: " + str(total_match))
        print("Total Different: " + str(total_different))
        print("Total Right: " + str(total_match + total_different))
        print("Total Wrong: " + str(total_wrong))
        print("Total Comparisons: " + str(total_comparisons))    
        print("Accuracy: " + str( format((total_match + total_different)/total_comparisons, '.2f') ))  
        print("=====================================================")

        total_match = 0;
        total_different = 0;
        total_comparisons = 0;
        total_wrong = 0;

        values = [4000, 0.1, 0.004, 30]
        globals()['img_dict'] = {}
        globals()['compared_img_list'] = []
        test_system(sim_name="m33_realism", galaxy_name="m31" , thresholds=values, sharpen=True, show_image=False, location='./') 

        print("=====================================================")
        print("Comparing simM33 to realM31")
        print("Total Matches: " + str(total_match))
        print("Total Different: " + str(total_different))
        print("Total Right: " + str(total_match + total_different))
        print("Total Wrong: " + str(total_wrong))
        print("Total Comparisons: " + str(total_comparisons))   
        print("Accuracy: " + str( format((total_match + total_different)/total_comparisons, '.2f') ))   
        print("=====================================================")

        total_match = 0;
        total_different = 0;
        total_comparisons = 0;
        total_wrong = 0;

        values = [4000, 0.1, 0.004, 30]
        globals()['img_dict'] = {}
        globals()['compared_img_list'] = []
        test_system(sim_name="m81_realism", galaxy_name="m33" , thresholds=values, sharpen=True, show_image=False, location='./') 

        print("=====================================================")
        print("Comparing simM81 to realM33")
        print("Total Matches: " + str(total_match))
        print("Total Different: " + str(total_different))
        print("Total Right: " + str(total_match + total_different))
        print("Total Wrong: " + str(total_wrong))
        print("Total Comparisons: " + str(total_comparisons)) 
        print("Accuracy: " + str( format((total_match + total_different)/total_comparisons, '.2f') ))     
        print("=====================================================")


if __name__ == "__main__":
    # execute only if run as a script
    main()
