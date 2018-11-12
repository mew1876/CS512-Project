import random
import itertools
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def edgeDetection(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	bilateral = cv2.bilateralFilter(gray, 25, 300, 300)
	canny = cv2.Canny(bilateral, 25, 15)
	return canny

def getSaliency(image):
	static = cv2.saliency.StaticSaliencySpectralResidual_create()
	success, saliencyMap = static.computeSaliency(image)
	saliency = (saliencyMap * 255).astype(np.uint8)
	return saliency

def createMesh(canny, saliency):
	edgeChance = 1
	nonEdgeChance = 0.001
	edgeSuppressDistance = 10
	extraSuppressDistance = 30
	validEdgePoints = np.zeros(canny.shape, dtype=np.uint8)
	validExtraPoints = np.zeros(canny.shape, dtype=np.uint8)
	rows = canny.shape[0]
	cols = canny.shape[1]
	mesh = [(0, 0), (0, cols - 1), (rows - 1, cols - 1), (rows - 1, 0)]
	# Create Boundary Points
	for row in range(0, rows, 30):
		mesh.extend(((row, 0), (row, cols - 1)))
	for col in range(0, cols, 30):
		mesh.extend(((0, col), (rows - 1, col)))
	numBoundaryPoints = len(mesh)
	# Create Interior Points
	for row in range(rows):
		for col in range(cols):
				if canny[row, col] == 255:
					if validEdgePoints[row, col] == 0 and random.random() < edgeChance:
						mesh.append((row, col))
						distance = edgeSuppressDistance if saliency[row, col] == 0 else int(edgeSuppressDistance * 0.75)
						cv2.circle(validEdgePoints, (col, row), distance, (255), thickness=cv2.FILLED)
						cv2.circle(validExtraPoints, (col, row), extraSuppressDistance, (255), thickness=cv2.FILLED)
				else:
					if validExtraPoints[row, col] == 0 and random.random() < nonEdgeChance:
						mesh.append((row, col))
						cv2.circle(validExtraPoints, (col, row), extraSuppressDistance, (255), thickness=cv2.FILLED)
	return	np.array(mesh), numBoundaryPoints

def clusterPoints(mesh):
	clustering = DBSCAN(eps=30, min_samples=5).fit(mesh)
	return clustering

def drawtriangles(image, mesh):
	rect = (0, 0, image.shape[1], image.shape[0])
	subdiv = cv2.Subdiv2D(rect);
	subdiv.insert([point[::-1] for point in mesh])
	edgeList = subdiv.getEdgeList();
	size = image.shape
	for edge in edgeList:
	    pt1 = (edge[0], edge[1])
	    pt2 = (edge[2], edge[3])
	    # if rect contains point 1 and point 2 and point 3
	    if not pt1[0] < rect[0] and not pt1[1] < rect[1] and not pt1[0] > rect[2] and not pt1[1] > rect[3]:
	    	if not pt2[0] < rect[0] and not pt2[1] < rect[1] and not pt2[0] > rect[2] and not pt2[1] > rect[3]:
			        cv2.line(image, pt1, pt2, (0,0,0), 1, cv2.LINE_8, 0)

def drawPoints(image, mesh, clustering, numBoundaryPoints):
	colors = [(0,0,0)]
	for i in range(len(clustering.core_sample_indices_)):
		color = np.uint8([[[random.randint(0, 360), 255, 255]]])
		color = tuple(cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0,0].tolist())
		colors.append(color)
	for i, (row, col) in enumerate(mesh):
		color = colors[clustering.labels_[i - numBoundaryPoints] + 1] if i >= numBoundaryPoints else (255, 255, 255)
		cv2.circle(image, (col, row), 3, color, thickness=cv2.FILLED)

def getEdgeLengths(mesh):
	edgeLengths = []
	for i in range(len(mesh)):
		edgeILengths = []
		for j in range(len(mesh)):
			edgeILengths.append(((mesh[i, 0] - mesh[j, 0]) ** 2 + (mesh[i, 1] - mesh[j, 1]) ** 2) ** 0.5)
		edgeLengths.append(edgeILengths)
	return edgeLengths

def getClusterAreas(clustering):
	pass

def graphMatch(sourceMesh, sourceClustering, referenceMesh, referenceClustering):
	# Get cluster areas
	
	# Get edge lengths
	sourceEdges = getEdgeLengths(sourceMesh)
	refEdges = getEdgeLengths(referenceMesh)
	# Create affinity matrix
	dimension = sourceMesh.shape[0] * referenceMesh.shape[0]
	affinity = np.ndarray(shape=(dimension, dimension), dtype=float)
	for row in range(dimension):
		sourceIndex1 = row // referenceMesh.shape[0]
		refIndex1 = row % referenceMesh.shape[0]
		for col in range(dimension):
			sourceIndex2 = col // sourceMesh.shape[0]
			refIndex2 = col % sourceMesh.shape[0]
			if sourceIndex1 == sourceIndex2 and refIndex1 == refIndex2:
				affinity[row, col] = min(sourceMesh[sourceIndex1][0], referenceMesh[refIndex1][0]) / max(sourceMesh[sourceIndex1][0], referenceMesh[refIndex1][0])
			elif sourceIndex1 != sourceIndex2 and refIndex1 != refIndex2:
				affinity[row, col] = 1 / abs(sourceEdges[sourceIndex1][sourceIndex2] - refEdges[refIndex1][refIndex2])
			else:
				affinity[row, col] = 0
	# Find best matching
	bestMatch = []
	bestScore = 0
	for indices in itertools.permutations(range(referenceMesh.shape[0])):
		matchVector = []
		for index in indices:
			match = [0] * referenceMesh.shape[0]
			match[index] = 1
			matchVector.extend(match)
		matchVector = np.array(matchVector)
		score = matchVector.dot(affinity.dot(np.transpose(matchVector)))
		if score > bestScore:
			bestScore = score
			bestMatch = indices
	return bestMatch

def processImage(image):
	canny = edgeDetection(image)
	saliency = getSaliency(image)
	_,  processedSaliency = cv2.threshold(saliency, 30, 255, cv2.THRESH_BINARY)
	mesh, numBoundaryPoints = createMesh(canny, processedSaliency)
	meshWithoutBoundaries = np.array(mesh[numBoundaryPoints:])
	clustering = clusterPoints(meshWithoutBoundaries)
	meshImage = image.copy()
	drawtriangles(meshImage, mesh)
	drawPoints(meshImage, mesh, clustering, numBoundaryPoints)
	return mesh, clustering, numBoundaryPoints, meshImage

def main():
	# sourceName = '../beach.jpg'
	sourceName = '../Toss.png'
	referenceName = '../volley.png'
	# sourceName = '../cat.png'
	# sourceName = '../elephant.png'
	# sourceName = '../right tree.png'
	# sourceName = '../center tree.png'
	# sourceName = '../lego.png'

	source = cv2.imread(sourceName)
	reference = cv2.imread(referenceName)
	sourceMesh, sourceClustering, sourceNumBoundaryPoints, sourceMeshImage = processImage(source)
	referenceMesh, referenceClustering, referenceNumBoundaryPoints, referenceMeshImage = processImage(reference)

	cv2.imshow('SourceMesh', sourceMeshImage)
	cv2.imshow('ReferenceMesh', referenceMeshImage)
	cv2.waitKey(0)

main()