import math
import numpy as np

IMAGE_HEIGHT = 684
IMAGE_WIDTH = 489
HOLES_N = 4
CUT_DISTANCE = 25
HOLE = 0
FINGER = 1

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def setmXmY(array):
    global mx
    global my
    mx = sum(x[0] for x in array) / len(array)
    my = sum(x[1] for x in array) / len(array)

def sort_clockwise(vect):
    return (math.atan2(vect[0] - mx, vect[1] - my) + 2 * math.pi) % (2*math.pi)

def deleteSimilar(array, indexArray = None):

    N = len(array)
    i = 0
    while i < N - 1:
        first = array[i]
        second = array[i + 1]
        if first[0] == second[0] and first[1] == second[1]:
            del array[i]
            if indexArray != None:
                del indexArray[i]
            N -= 1
        else:
            i += 1

    if indexArray != None:
        return array, indexArray
    else:
        return array

def findShift(array):

    N = len(array)
    routeLenMin = IMAGE_HEIGHT * IMAGE_WIDTH
    shiftFound = 0

    for shift in xrange(N):
        routeLenCurrent = 0
        for i in xrange(N - 1):
            distance = length(array[(i + shift) % N] - array[(i + 1 + shift) % N])
            routeLenCurrent += distance
        if routeLenCurrent < routeLenMin:
            routeLenMin = routeLenCurrent
            shiftFound = shift

    return shiftFound

def clockwisePoints(points):
    pointsSort = []
    types = []

    for i in xrange(len(points)):
        pointsSort.append(points[i][0])
        pointsSort.append(points[i][2])
        pointsSort.append(points[i][1])
        types += [FINGER, HOLE, FINGER]

    for i in xrange(HOLES_N):
        first = pointsSort[3 * i]
        distance = IMAGE_HEIGHT * IMAGE_WIDTH
        position = -1
        found_element = None
        for j in xrange(HOLES_N):
            second = pointsSort[3 * j + 2]
            cur_dist = length(first - second)
            if cur_dist < distance:
                distance = cur_dist
                position = j
                found_element = second
        if distance < CUT_DISTANCE:
            pointAvg = (first + found_element) / 2
            pointsSort[3 * i] = pointAvg
            pointsSort[3 * position + 2] = pointAvg

    pointsSort, types = deleteSimilar(pointsSort, types)
    fingers = []
    holes = []

    for i in xrange(len(pointsSort)):
        point = pointsSort[i]
        if types[i] == HOLE:
            holes.append(point)
        else:
            fingers.append(point)

    setmXmY(fingers)
    fingers.sort(key=sort_clockwise)
    setmXmY(holes)
    holes.sort(key=sort_clockwise)
    fingers = deleteSimilar(fingers)
    shift_holes = findShift(holes)
    shift_fingers = findShift(fingers)
    pointsSort = []

    for i in xrange(HOLES_N):
        pointsSort.append(fingers[(i + shift_fingers) % 5])
        pointsSort.append(holes[(i + shift_holes) % HOLES_N])
    pointsSort.append(fingers[(4 + shift_fingers) % 5])

    return pointsSort

def getPointsContour(defects, contour):
    pointsAngle = []
    # we wanna delete wrong defects
    # delete holes that form big angles with hull
    N = defects.shape[0]
    i = 0
    while i < N:
        s, e, f, d = defects[i, 0]
        start = contour[s][0]
        end = contour[e][0]
        far = contour[f][0]
        cos = angle(start - far, end - far)
        if cos < math.pi / 2 + 0.05:
            pointsAngle.append([start, end, far])
        i += 1
    pointsAngle = np.array(pointsAngle)
    # delete holes near hull
    if len(pointsAngle) > HOLES_N:
        pointsDistance = []
        i = 0
        distances = []
        while i < len(pointsAngle):
            start = pointsAngle[i][0]
            end = pointsAngle[i][1]
            far = pointsAngle[i][2]
            cos = math.cos(angle(start - far, start - end))
            distance = math.sqrt(1 - cos * cos) * length(start - far)
            distances.append(distance)
            i += 1
        pointsDistance.append(pointsAngle[np.argsort(distances)[-4:].tolist()])
        pointsAngle = pointsDistance[0]

    return pointsAngle