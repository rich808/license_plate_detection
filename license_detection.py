# import libraries 
import cv2
import imutils
import pytesseract
import numpy as np
import matplotlib.pyplot as plt


# located tesseract cmd
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


# read in the image
lp = cv2.imread('./us_license_plate.jpg')

# standardized the image to width 512
lp = imutils.resize(lp, width = 512)

# convert to gray scale. *not only this step reduce computation complexity, it is also important for finding contour because
# openCV finds contour from white connecting objects in black background. 

lp_bw = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)


# use canny function to detect all the edges then use findcontour to find all the contours
# three inputs. (input_soure, minVal, maxVal).minVal and maxVal is the threshold value to indicate the edges using
# the L2Gradient method (thin vs thick edges). Edge is determine from change in gradient (first derivatives). 
# L2Gradient is the method use to calculate gradient intensity. If True, it uses pythagorean theorem. If False (default),
# it add the magnitude of x and y gradient


lp_bw = cv2.bilateralFilter(lp_bw, 10, 50, 50)

lp_canny = cv2.Canny(lp_bw, 100, 200, L2gradient = False)


# find the contour
contours, hier = cv2.findContours(lp_canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# output is contour and hierarchy
# contour is a list of np.array of (x,y) coordinates

# since it created many contours in the image, the computer needs to know which contours will be the license plate. 
# We can filter it by using the area. Filter the areas to find the license plate

# use the area to find the contour. Sort the top 15 contours with the biggest area
# need to reverse the 'sorted' function to have it start with the biggest contour
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:15]

# find and locate the contours of the license plate

# Need to iterate through each of the contours to find it

# set a variable to store the license plate once the contour is found. 
lp_contour = None

# iterate through each of the contours
for contour in contours:
    
    # for each contour, find its perimeters, using cv2.arclength(contour, True/False). If true, closed contour. False, curve
    arclength = cv2.arcLength(contour, True)
    
    # since the contour is not a perfect shape, we will need to approximate the shape by giving a maximum distance of the 
    # contour to still be considered as a contour.
    # give it 1% of the arclength for the approximate distance
    approx_dist = 0.01 * arclength
    
    # approximate a closed contour. 
    # first parameter is the contour, second is the maximum distance, third if True is closed or false is open contour
    approx_contour = cv2.approxPolyDP(contour, approx_dist, True)
    
    # from the approximated contour, if it is a closed contour and rectangular shape, it will have 4 corners. Let's check it
    if len(approx_contour) == 4:
        # if we have 4 corners, then we have a shape of an license plate. Store it to the lp_contour as the license plate
        lp_contour = approx_contour
        # form a rectangular countour
        # since is a rectangle, use boundingRect. (x,y) is the bottom left, w (width), h(height) to locate the points
        x, y, w, h = cv2.boundingRect(contour)
    
        # crop this image from the original image
        # x from x to x + w as the full width
        # y from y to y + h as the full height
        lp_bw_crop = lp_bw[y:y+h, x:x+w]
        
        # finish the loop once it has identified this
        break



# use pytesseract to extract the text from the image

lp_text = pytesseract.image_to_string(lp_bw_crop, config='--psm 10')[:7]

# put text into the original image
img_text = cv2.putText(lp, lp_text, (x, y+105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)


# display the image

cv2.imshow('with_license_number', img_text)

# 'escape' == 27. If 'escape', then closed all windows
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()


