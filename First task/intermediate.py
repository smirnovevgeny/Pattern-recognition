import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import numpy as np
import cv2
import math
import collections
import os
import argparse

#set default paremeters
SHOW_MASK_RED = False
SHOW_MASK_BLUE = False
SHOW_MASK_YELLOW = False
SHOW_MAIN_CONTOURS = False

#parse arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--show_mask',help = "masks to show",default="")
parser.add_argument("--folder",help = "path to folder to search files", default="Sample")
parser.add_argument("-f","--files", help="files to use", default='Group*')
parser.add_argument("-l", '--show_labels', help="to show labels of edges", default=False, action='store_true')
parser.add_argument("-n", '--not_show_images', help="to not show images", default=False, action='store_true')
parser.add_argument('--show_all_contours', help='to show all contours after first step', action='store_true', default=False)
parser.add_argument("--show_main_contours", help = 'to show only chips contours', action='store_true', default=False)
parser.add_argument("--show_edges_filter", help = 'to show edges filter', action='store_true', default=False)
parser.add_argument("-s","--save_images", help = "save images", action='store_true', default=False)
args = parser.parse_args()

if "r" in args.show_mask:
    SHOW_MASK_RED = True
if "b" in args.show_mask:
    SHOW_MASK_BLUE = True
if "y" in args.show_mask:
    SHOW_MASK_YELLOW = True

#choose file to read
f = os.popen("ls "+args.folder+" | grep "+args.files + " | grep '.bmp'")
ls = f.read()
ls = ls.split("\n")
objects = [args.folder+"/" + x for x in ls]
objects = objects[:-1]
train = ["Sample/Dozen_0.bmp"]

#red filter
def red_filter(img):
    #find colour in range
    boundaries = [
        ([0, 0, 90], [61, 56, 255])
    ]
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)
        output = np.array(cv2.split(output)[2])
        output[output!=0] = 255
    return output

#yellow filter
def y_filter(img):
    #get green component of image
    img = cv2.split(img)[1]
    #smooth image
    img = cv2.medianBlur(img,5)
    #binarize image
    img = np.array(cv2.threshold(img, 95, 255, cv2.THRESH_TOZERO_INV)[1])
    zero = img != 0
    one = img == 0
    img[one] = 255
    img[zero] = 0
    return img

#blue filter
def blue_filter(img):
    # find colour in range
    boundaries = [
        ([39, 40, 41], [125, 150, 150])
    ]
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)
        output = np.array(cv2.split(output)[1])
        output[output != 0] = 255
    return output

#another blue filter
def blue2_filter(img):
    #get blue component
    img = cv2.split(img)[0]
    #binarize image
    img = np.array(cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO_INV)[1])
    zero = img != 0
    one = img == 0
    img[one] = 255
    img[zero] = 0
    return img

    return output

#emphasize contour
def emp_contour(image, msize1, msize2, number):
    #smooth image
    th2 = cv2.medianBlur(image, msize1)
    th2 = cv2.adaptiveThreshold(th2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 1)
    #delete contact areas
    th2 = cv2.erode(th2, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    # smooth image
    th2 = cv2.medianBlur(th2, msize2)
    if args.show_edges_filter:
        cv2.imshow("finding edges",th2)
        if args.save_images:
            cv2.imwrite("find_edges_"+args.files+"_"+str(number)+".jpg",th2)
        cv2.waitKey(0)
    return th2

#read all images
def read(objects):

    images_grey = []
    images_color = []
    mask_red = []
    mask_y = []
    mask_blue = []
    mask_blue2 = []
    images = []

    for file in objects:
        img = cv2.imread(file, 0)
        images_grey.append(img)
        images.append(img)
        img = cv2.imread(file, 1)
        #get red, blue, yellow, masks
        mask_red.append(red_filter(img))
        mask_y.append(y_filter(img))
        mask_blue.append(blue_filter(img))
        mask_blue2.append(blue2_filter(img))
        images_color.append(img)

    adaptive_mean = []

    for k, image in enumerate(images):
        #prepare images to find contours
        adaptive_mean.append(emp_contour(image,5,5,k))
    return adaptive_mean, images, mask_red, mask_blue, mask_blue2, mask_y, images_color

#get contours from image before filtered
def getContours(array):
    contours = []
    cont_area_mins = []
    for image in array:
        #find all images contours
        contours.append(cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])
        #find only external contours
        external = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        #find min area of external contours
        cont_area_min =np.max([cv2.contourArea(x) for x in external])*0.75
        cont_area_mins.append(cont_area_min)
    return contours, cont_area_mins



def main(learn_mode, contours, cont_area_mins, images_color, clfs = None):
    # learn mode to teach classifiers
    if learn_mode:
        array_red =[]
        array_blue = []
        array_blue2 = []
        array_y = []
        number = []
    for k in range(len(images)):
        areas = []
        for cnt in contours[k]:
            area = cv2.contourArea(cnt)
            if area < cont_area_mins[k] :
                areas.append(area)
        areas = np.array(areas)
        #we wanna to delete contours with area less than the half of the maximum non external
        thresh = 0.5*areas.max()
        cont = []
        cnt_n = 0
        for cnt in contours[k]:
            area = cv2.contourArea(cnt)
            if thresh < area < cont_area_mins[k]:
                #approximate with convex
                approx = cv2.convexHull(cnt)
                # use are dependency approximation
                epsilon = 0.05*thresh/area * cv2.arcLength(approx, True)
                approx = cv2.approxPolyDP(approx, epsilon, True)
                #estimate centers of contours
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if len(approx) == 6:
                    if not learn_mode:
                        descr = ["" for x in range(len(clfs))]
                    for p in range(6):
                        red = mask_red[k].copy()
                        blue = mask_blue[k].copy()
                        blue2 = mask_blue2[k].copy()
                        yellow = mask_y[k].copy()
                        x1 = approx[p % 6][0][0]
                        x2 = approx[(p+1)%6][0][0]
                        y2 = approx[(p+1)%6][0][1]
                        y1 = approx[p%6][0][1]
                        dx = x2 - x1
                        dy = y2 - y1
                        #get an angle
                        rads = math.atan2(dy, dx)
                        rads %= 2 * math.pi
                        degs = math.degrees(rads)
                        centerPoint = ((x1+x2)/2,(y1+y2)/2)
                        #get length of edge
                        rad = int(math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)))
                        mask = np.zeros(images[k].shape)
                        cv2.ellipse(mask, centerPoint, (rad/3, int(rad/3)), degs, 30, 150, 1, -1)
                        ellipse_area = np.sum(mask!=0)
                        #get the intersections of ellipse and color masks
                        red[mask==0] = 0
                        blue[mask == 0] = 0
                        blue2[mask == 0] = 0
                        yellow[mask == 0] = 0
                        area_red = np.sum(red!=0)
                        area_blue2 = np.sum(blue2!=0)
                        area_blue = np.sum(blue != 0)
                        area_y = np.sum(yellow!=0)
                        #estimate the part of mask filled by special color
                        q_red = round(float(area_red)/ellipse_area,2)
                        q_blue = round(float(area_blue)/ellipse_area,2)
                        q_blue2 = round(float(area_blue2) / ellipse_area, 2)
                        q_y = round(float(area_y)/ellipse_area,2)
                        if learn_mode:
                            array_red.append(q_red)
                            array_blue.append(q_blue)
                            array_blue2.append(q_blue2)
                            array_y.append(q_y)
                        if not learn_mode:
                            text=""
                            #predict the color
                            for i in range(len(clfs)):
                                if i<3:
                                    descr[i]+=str(clfs[i].predict(np.array([q_y,q_blue, q_red]).reshape(1, -1))[0])
                                else:
                                    descr[i] += str(clfs[i].predict(np.array([q_y, q_blue2, q_red]).reshape(1, -1))[0])
                                text+=descr[i][-1]
                            if args.show_labels:
                                cv2.putText(images_color[k], text, centerPoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,0)
                    if not learn_mode:
                        #if predicted colors doesn't give us the one of chins than use next
                        for i in range(len(clfs)):
                            s = descr[i]
                            text = ""
                            for p in range(10):
                                if s in perms[p]:
                                    text = str(numbers[p])
                                    break;
                            if text != "":
                                cv2.putText(images_color[k], text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (255, 255, 255), 2, 0)
                                break;
                    cnt_n+=1
                    cont.append(approx)
        contours[k] = cont
        if not learn_mode:
            if args.show_main_contours:
                cv2.drawContours(images_color[k], contours[k], -1, (255, 255, 0), 2)
            if not args.not_show_images:
                cv2.imshow("color", images_color[k])
                if args.save_images:
                    cv2.imwrite("images_" + args.files + "_" + str(k) + ".jpg", images_color[k])
            if SHOW_MASK_RED:
                cv2.imshow("red", mask_red[k])
                if args.save_images:
                    cv2.imwrite("mask_red_" + args.files + "_" + str(k) + ".jpg", mask_red[k])
            if SHOW_MASK_BLUE:
                cv2.imshow("blue", mask_blue2[k])
                if args.save_images:
                    cv2.imwrite("mask_blue_" + args.files + "_" + str(k) + ".jpg", mask_blue[k])
            if SHOW_MASK_YELLOW:    
                cv2.imshow("yellow", mask_y[k])
                if args.save_images:
                    cv2.imwrite("mask_yellow_" + args.files + "_" + str(k) + ".jpg", mask_y[k])
            if not args.not_show_images or  SHOW_MASK_BLUE or  SHOW_MASK_RED or SHOW_MASK_YELLOW:
                cv2.waitKey(0)
        else:
            return array_blue, array_blue2, array_red, array_y,number


adaptive_mean, images, mask_red, mask_blue, mask_blue2, mask_y, images_color = read(train)
contours, cont_area_mins = getContours(adaptive_mean)
blue, blue2, red, yellow,number = main(True,contours,cont_area_mins,images_color)

#prepare training sample
ans = list("yybrbrrrbyybyyrbrbbyrybrrbbyyrbbryyrbbryrybryrbyrybyrbbbyryr")
perms = []
numbers = [1,2,10,9,3,5,8,6,4,7]
for k in range(10):
    perm =[]
    seq = np.array((ans[6*k:6*(k+1)]))
    perm.append("".join(seq))
    d = collections.deque(seq)
    for k in range(5):
        d.rotate(1)
        perm.append("".join(d))
    perms.append(perm)

import pandas as pd
data = pd.DataFrame(ans)
data["yellow"] = yellow
data["blue"] = blue
data["blue2"] = blue2
data["red"] = red
y = data[0]
X1 = data[["yellow","blue","red"]]
X2 = data[["yellow","blue2","red"]]

#train classifiers
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression()
clf1.fit(X1,y)
from sklearn.ensemble import GradientBoostingClassifier
clf2 = GradientBoostingClassifier()
clf2.fit(X1,y)
from sklearn.neighbors import KNeighborsClassifier
clf3 = KNeighborsClassifier()
clf3.fit(X1,y)

from sklearn.linear_model import LogisticRegression
clf4 = LogisticRegression()
clf4.fit(X2,y)
from sklearn.ensemble import GradientBoostingClassifier
clf5 = GradientBoostingClassifier()
clf5.fit(X2,y)
from sklearn.neighbors import KNeighborsClassifier
clf6 = KNeighborsClassifier()
clf6.fit(X2,y)


adaptive_mean, images, mask_red, mask_blue, mask_blue2, mask_y, images_color = read(objects)
contours, cont_area_mins = getContours(adaptive_mean)

#print contours
if args.show_all_contours:
    for k in range(len(images)):
        free = np.zeros(images_color[k].shape)
        cv2.drawContours(free, contours[k], -1, (255, 255, 0), 2)
        if args.save_images:
            cv2.imwrite("all_contours_" + args.files + "_" + str(k) + ".jpg", free)
        cv2.imshow('',free)
        cv2.waitKey(0)

main(False,contours,cont_area_mins,images_color, [clf1,clf2,clf3,clf4,clf5,clf6])