import random
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# fileName = '../beach.jpg'
fileName = '../Toss.png'

original = cv2.imread(fileName)

# Edge Detection
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 20000, 500, apertureSize=7)

# Saliency
static = cv2.saliency.StaticSaliencySpectralResidual_create()
success, saliencyMap = static.computeSaliency(original)
saliency = (saliencyMap * 255).astype("uint8")

# Create Mesh Points
edgeChance = 0.1
nonEdgeChance = 0.001
edgeSuppressDistance = 10
extraSuppressDistance = 30
points = original.copy()
validEdgePoints = np.zeros(canny.shape, dtype=np.uint8)
validExtraPoints = np.zeros(canny.shape, dtype=np.uint8)
meshPoints = []
rect = (0, 0, original.shape[1], original.shape[0])
subdiv = cv2.Subdiv2D(rect);
for row in range(points.shape[0]):
	for col in range(points.shape[1]):
			if canny[row, col] == 255:
				if validEdgePoints[row, col] == 0 and random.random() < edgeChance:
					meshPoints.append([row, col])
					cv2.circle(points, (col, row), 3, (0, 255, 0), thickness=cv2.FILLED)
					cv2.circle(validEdgePoints, (col, row), int(edgeSuppressDistance), (255), thickness=cv2.FILLED)
					cv2.circle(validExtraPoints, (col, row), extraSuppressDistance, (255), thickness=cv2.FILLED)
					subdiv.insert((col,row))
			else:
				if validExtraPoints[row, col] == 0 and random.random() < nonEdgeChance:
					meshPoints.append([row, col])
					cv2.circle(points, (col, row), 3, (255, 0, 0), thickness=cv2.FILLED)
					cv2.circle(validExtraPoints, (col, row), extraSuppressDistance, (255), thickness=cv2.FILLED)
					subdiv.insert((col,row))

# Group Points
mesh = np.array(meshPoints)

# Draw Delaunay Triangulation 
edgeList = subdiv.getEdgeList();
size = original.shape
triangulation = original.copy()
for edge in edgeList:
    pt1 = (edge[0], edge[1])
    pt2 = (edge[2], edge[3])
    # if rect contains point 1 and point 2 and point 3
    if not pt1[0] < rect[0] and not pt1[1] < rect[1] and not pt1[0] > rect[2] and not pt1[1] > rect[3]:
    	if not pt2[0] < rect[0] and not pt2[1] < rect[1] and not pt2[0] > rect[2] and not pt2[1] > rect[3]:
		        cv2.line(triangulation, pt1, pt2, (0,0,0), 1, cv2.LINE_8, 0)

# cv2.imshow('Original', original)
# cv2.imshow('Canny', canny)
# cv2.imshow('Saliency', saliency)
cv2.imshow('Points', points)
cv2.imshow('Triangulation', triangulation)
cv2.waitKey(0)