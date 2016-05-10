import argparse
import os
import numpy as np
import cv2
from Contours import  *
from Tools import *
from Clustering import *

IMAGES_N = 0
FEATURES_N = 12
FOLDER_CONTOURS_ALL = "Contours_all"
FOLDER_CONTOURS_MAIN = "Contours_main"
ENDING_CONTOURS = "_contours"
FOLDER_RESULT_LINES = "Result_lines"
ENDING_LINES = "_lines"
FOLDER_EDGES = "Edges"
ENDING_EDGES = "_edges_filter"
FILE_EXTENSION = ".jpg"

imagesColor = []
imagesGrey = []
imagesContour = []

# parse arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--show_mask', help="masks to show", default="")
parser.add_argument("--folder", help="path to folder to search files", default="training")
parser.add_argument("-f", "--files", help="files to use", default='.tif')
parser.add_argument("-l", '--show_labels', help="to show labels of edges", default=False, action='store_true')
parser.add_argument("-n", '--not_show_images', help="to not show images", default=False, action='store_true')
parser.add_argument('--show_all_contours', help='to show all contours after first step', action='store_true',
                    default=False)
parser.add_argument("--show_main_contours", help='to show only chips contours', action='store_true', default=False)
parser.add_argument("--show_lines", help='to show result lines', action='store_true', default=False)
parser.add_argument("--show_edges_filter", help='to show edges filter', action='store_true', default=False)
parser.add_argument("-s", "--save_images", help="save images", action='store_true', default=True)
parser.add_argument("--select_clusters_n", help="show graphs to select number of clusters", action='store_true', default=False)
args = parser.parse_args()

# choose file to read
f = os.popen("ls " + args.folder + " | grep " + args.files + " | grep '.tif'")
ls = f.read()
ls = ls.split("\n")
objects = [args.folder + "/" + x for x in ls]
objectsNames = [x for x in ls]
objectsNames = objectsNames[:-1]
objects = objects[:-1]


def emphasizeContour(image, number):
    # Filter for contour emphasizing 
    image = cv2.medianBlur(image, 5)
    threshold = cv2.threshold(image, 45, 255, cv2.THRESH_TOZERO_INV)[1]
    threshold  = cv2.adaptiveThreshold(threshold,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,5,2)
    if args.show_edges_filter:
        cv2.imshow("Edges Filter", threshold)
        if args.save_images:
            cv2.imwrite(FOLDER_EDGES + "/" +objectsNames[number] + ENDING_EDGES + FILE_EXTENSION, threshold)
        cv2.waitKey(0)

    return threshold

def readImages(objects):
    # read images
    global IMAGES_N
    global imagesColor
    global imagesGrey
    global imagesContour
    IMAGES_N = len(objects)

    for i, path in enumerate(objects):
        image = cv2.imread(path, 0)
        imagesGrey.append(image)

        image = emphasizeContour(image, i)
        imagesContour.append(image)
        image = cv2.imread(path, 1)
        imagesColor.append(image)

def showContours(contours, images, folder):
    # show contours after filtering
    for k in xrange(len(contours)):
        free = np.array(images[k])
        cv2.drawContours(free, contours[k], -1, (255, 255, 0), 2)
        if args.save_images:
            cv2.imwrite(folder+ "/" + objectsNames[k] + ENDING_CONTOURS + FILE_EXTENSION, free)
        if args.show_all_contours:
            cv2.imshow('', free)
            cv2.waitKey(0)


def getFeatures(contoursClean):
    # get features for feature clustering
    X = np.zeros((IMAGES_N, FEATURES_N))
    for k in xrange(IMAGES_N):
        contour = contoursClean[k]
        image = imagesColor[k]
        hull = cv2.convexHull(contour, clockwise=True, returnPoints = False)
        # holes is the poins between fingers
        # wanna find all defects and collect only holes
        defects = cv2.convexityDefects(contour, hull)

        pointsContour = getPointsContour(defects, contour)
        pointsSort = clockwisePoints(pointsContour)
        # add distances between holes as features
        for i in xrange(4):
            X[k, 8 + i] = length(pointsSort[2 * i % 4] - pointsSort[2 * (i + 1) % 4])
        # show lines and add features
        for i in xrange(len(pointsSort) - 1):
            start = pointsSort[i]
            end = pointsSort[i + 1]
            distance = length(start - end)
            # add distances between hole and nearest fingers as features
            X[k,i] = distance
            if args.save_images:
                cv2.line(image, tuple(start), tuple(end), [0, 255, 0], 2)
                cv2.imwrite(FOLDER_RESULT_LINES + "/" + objectsNames[k] + ENDING_LINES + FILE_EXTENSION, image)
            if args.show_lines:
                cv2.line(image, tuple(start), tuple(end), [0, 255, 0], 2)
                cv2.imshow('image', image)
                cv2.waitKey(0)

    return X

readImages(objects)
contours, contourAreaMins = getContours(imagesContour)
showContours(contours, imagesColor, FOLDER_CONTOURS_ALL)
contoursClean = getContoursClean(contours, contourAreaMins)
showContours(contoursClean, imagesColor, FOLDER_CONTOURS_MAIN)
X = getFeatures(contoursClean)

if args.select_clusters_n:
    showKMeans(X, IMAGES_N)
    showMiniBatchKMeans(X, IMAGES_N)

writeKmeans(X, KMEANS_N, objectsNames)
writeMiniBatchKMeans(X, MINIBATCHKMEANS_N, objectsNames)
writeSpectralClustering(X, SPECTRALCLUSTERING_N, objectsNames)