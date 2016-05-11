# methods to find contours
import cv2
import numpy as np

def getContoursClean(contours, contourAreaMins):

    N = len(contours)
    contoursClean = []
    for image in xrange(N):
        areas = []
        for contour in contours[image]:
            area = cv2.contourArea(contour)
            if area < contourAreaMins[image]:
                areas.append(area)
        areas = np.array(areas)
        # we wanna to delete contours with area less than the half of the maximum non external
        thresh = 0.5 * areas.max()
        contourClean = []
        contorCleanPerimeter = []
        for contour in contours[image]:
            area = cv2.contourArea(contour)
            if thresh < area < contourAreaMins[image]:
                epsilonRectangle = 5
                approximateRectangle = cv2.approxPolyDP(contour, epsilonRectangle, True)
                # delete rectangle contours
                if len(approximateRectangle) != 4:
                    contourClean.append(contour)
                    contorCleanPerimeter.append(len(contour))
        contorCleanPerimeter = np.array(contorCleanPerimeter)
        contourClean = contourClean[contorCleanPerimeter.argmax()]
        contoursClean.append(contourClean)

    return contoursClean

def getContours(images):

    contours = []
    contourAreaMins = []

    for image in images:
        # find all images contours
        contours.append(cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0])
        # find only external contours
        external = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # find min area of external contours
        contourAreaMin = np.max([cv2.contourArea(x) for x in external]) * 0.75
        contourAreaMins.append(contourAreaMin)
    return contours, contourAreaMins