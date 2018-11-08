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
saliency = (saliencyMap * 255).astype(np.uint8)
_,  processedSaliency = cv2.threshold(saliency, 30, 255, cv2.THRESH_BINARY)
processedSaliency = cv2.erode(processedSaliency, np.ones((15,15), np.uint8))

# Create Mesh Points
edgeChance = 0.1
nonEdgeChance = 0.001
edgeSuppressDistance = 10
extraSuppressDistance = 30
points = original.copy()
meshImage = original.copy()
validEdgePoints = np.zeros(canny.shape, dtype=np.uint8)
validExtraPoints = np.zeros(canny.shape, dtype=np.uint8)
rows = points.shape[0]
cols = points.shape[1]
meshPoints = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
# Create Boundary Points
for row in range(0, rows, 30):
	meshPoints.extend(((row, 0), (row, cols - 1)))
for col in range(0, cols, 30):
	meshPoints.extend(((0, col), (rows - 1, col)))
# Create Interior Points
for row in range(rows):
	for col in range(cols):
			if canny[row, col] == 255 or processedSaliency[row, col] == 255:
				if validEdgePoints[row, col] == 0 and random.random() < edgeChance:
					meshPoints.append((row, col))
					distance = edgeSuppressDistance if processedSaliency[row, col] == 0 else int(edgeSuppressDistance * 0.9)
					cv2.circle(validEdgePoints, (col, row), distance, (255), thickness=cv2.FILLED)
					cv2.circle(validExtraPoints, (col, row), extraSuppressDistance, (255), thickness=cv2.FILLED)
			else:
				if validExtraPoints[row, col] == 0 and random.random() < nonEdgeChance:
					meshPoints.append((row, col))
					cv2.circle(validExtraPoints, (col, row), extraSuppressDistance, (255), thickness=cv2.FILLED)

# Group Points
mesh = np.array(meshPoints)
clustering = DBSCAN(eps=25, min_samples=15).fit(mesh)
np.set_printoptions(threshold=np.nan)
# print(clustering.labels_)

# Draw Delaunay Triangulation 
rect = (0, 0, original.shape[1], original.shape[0])
subdiv = cv2.Subdiv2D(rect);
subdiv.insert([point[::-1] for point in meshPoints])
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
		        cv2.line(meshImage, pt1, pt2, (0,0,0), 1, cv2.LINE_8, 0)

# Draw points
colors = [(0,0,0)]
for i in range(len(clustering.core_sample_indices_)):
	color = np.uint8([[[random.randint(0, 360), 255, 255]]])
	color = tuple(cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0,0].tolist())
	colors.append(color)
for i, (row, col) in enumerate(mesh):
	cv2.circle(points, (col, row), 3, colors[clustering.labels_[i] + 1], thickness=cv2.FILLED)
	cv2.circle(meshImage, (col, row), 3, colors[clustering.labels_[i] + 1], thickness=cv2.FILLED)


# cv2.imshow('Original', original)
# cv2.imshow('Canny', canny)
# cv2.imshow('Saliency', saliency)
# cv2.imshow('Processed', processedSaliency)
# cv2.imshow('Points', points)
# cv2.imshow('Triangulation', triangulation)
cv2.imshow('Mesh', meshImage)
cv2.waitKey(0)