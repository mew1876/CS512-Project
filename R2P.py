import random
import itertools
import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
import Project

def edgeDetection(image):
	gray = image #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	bilateral = cv2.bilateralFilter(gray, 25, 600, 600)
	canny = cv2.Canny(bilateral, 38, 15)
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
	clustering = DBSCAN(eps=52, min_samples=17).fit(mesh)
	return clustering

def drawtriangles(image, mesh):
	rect = (0, 0, image.shape[1], image.shape[0])
	subdiv = cv2.Subdiv2D(rect)
	for point in mesh:
		if point[0] >= 0 and point[0] < image.shape[0]:
			if point[1] >= 0 and point[1] < image.shape[1]:
				subdiv.insert([point[::-1]])
	edgeList = subdiv.getEdgeList()
	size = image.shape
	for edge in edgeList:
	    pt1 = (edge[0], edge[1])
	    pt2 = (edge[2], edge[3])
	    # if rect contains point 1 and point 2 and point 3
	    if not pt1[0] < rect[0] and not pt1[1] < rect[1] and not pt1[0] > rect[2] and not pt1[1] > rect[3]:
	    	if not pt2[0] < rect[0] and not pt2[1] < rect[1] and not pt2[0] > rect[2] and not pt2[1] > rect[3]:
			        cv2.line(image, pt1, pt2, (0,0,0), 1, cv2.LINE_8, 0)

def drawPoints(image, mesh, clustering, clusteringInfo, numBoundaryPoints):
	colors = [(0,0,0)]
	for i in range(len(set(clustering.labels_)) - 1):
		color = np.uint8([[[25 * i, 255, 255]]])
		color = tuple(cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0,0].tolist())
		colors.append(color)
	done = []
	for i, (row, col) in enumerate(mesh):
		color = colors[clustering.labels_[i - numBoundaryPoints] + 1] if i >= numBoundaryPoints else (255, 255, 255)
		cv2.circle(image, (int(col), int(row)), 3, color, thickness=cv2.FILLED)
		# if i >= numBoundaryPoints:
		# 	print(clustering.labels_[i - numBoundaryPoints] + 1)
		# print(row, col, color)
	_, centers, _ = zip(*clusteringInfo)
	for index, center in enumerate(centers):
		cv2.circle(image, (int(center[1]), int(center[0])), 20, colors[index + 1], thickness=cv2.FILLED)
		# print((int(center[0]), int(center[1])), colors[index + 1])
		# print('c', center, colors[index + 1])

def getNormalizedEdgeLengths(clusteringInfo, shape):
	edgeLengths = []
	for i in range(len(clusteringInfo)):
		edgeILengths = []
		for j in range(len(clusteringInfo)):
			edgeILengths.append(((clusteringInfo[i][1][0] - clusteringInfo[j][1][0]) ** 2 + (clusteringInfo[i][1][1] - clusteringInfo[j][1][1]) ** 2) ** 0.5)
		edgeLengths.append(edgeILengths)
	return edgeLengths

def getClusterInfo(mesh, clustering, saliency):
	numClusters = len(set(clustering.labels_))
	clusters = []
	areas = []
	centers = []
	saliencyScores = []
	for i in range(numClusters):
		clusters.append([])
		saliencyScores.append(0)
	for index, label in enumerate(clustering.labels_):
		clusters[label + 1].append(mesh[index])
		saliencyScores[label + 1] += (saliency[mesh[index][0], mesh[index][1]])
	for index, cluster in enumerate(clusters):
		if len(cluster) == 0 or index == 0:
			continue
		contour = np.array(cluster)
		hull = cv2.convexHull(contour)
		moments = cv2.moments(contour)
		areas.append(cv2.contourArea(hull))
		# centers.append((moments['m10']/moments['m00'], moments['m01']/moments['m00']))
		centers.append(sum(contour) / len(contour))
	return list(zip(areas, centers, saliencyScores[1:]))

def reorder(clustering, clusteringInfo, reordering):
	reordering = dict(enumerate(reordering))
	for index, label in enumerate(clustering.labels_):
		if label > -1 and label < len(reordering):
			clustering.labels_[index] = reordering[label]
	newInfo = []
	inverseOrdering = [(v,k) for k,v in reordering.items()]
	inverseOrdering.sort(key=lambda x : x[0])
	inverseOrdering = dict(inverseOrdering)
	for key in inverseOrdering:
		newInfo.append(clusteringInfo[inverseOrdering[key]])
	return clustering, newInfo

def reorderBySaliency(clustering, clusteringInfo):
	_, _, saliencyScores = zip(*clusteringInfo)
	reordering = list(zip(range(len(clusteringInfo)), saliencyScores))
	reordering.sort(key=lambda x : x[1], reverse=True)
	reordering, _ = zip(*reordering)
	clustering, clusteringInfo = reorder(clustering, clusteringInfo, reordering)
	return clustering, clusteringInfo

def graphMatch(sourceClusteringInfo, sourceShape, referenceClusteringInfo, referenceShape):
	# Get edge lengths
	sourceEdges = getNormalizedEdgeLengths(sourceClusteringInfo, sourceShape)
	referenceEdges = getNormalizedEdgeLengths(referenceClusteringInfo, referenceShape)
	# Create affinity matrix
	dimension = len(sourceClusteringInfo) * len(referenceClusteringInfo)
	affinity = np.ndarray(shape=(dimension, dimension), dtype=float)
	for row in range(dimension):
		sourceIndex1 = row // len(referenceClusteringInfo)
		referenceIndex1 = row % len(referenceClusteringInfo)
		for col in range(dimension):
			sourceIndex2 = col // len(sourceClusteringInfo)
			referenceIndex2 = col % len(sourceClusteringInfo)
			if sourceIndex1 == sourceIndex2 and referenceIndex1 == referenceIndex2:
				affinity[row, col] = min(sourceClusteringInfo[sourceIndex1][0], referenceClusteringInfo[referenceIndex1][0]) / max(sourceClusteringInfo[sourceIndex1][0], referenceClusteringInfo[referenceIndex1][0])
			elif sourceIndex1 != sourceIndex2 and referenceIndex1 != referenceIndex2:
				affinity[row, col] = 1 / abs(sourceEdges[sourceIndex1][sourceIndex2] - referenceEdges[referenceIndex1][referenceIndex2])
			else:
				affinity[row, col] = 0
	# Find best matching
	bestMatch = []
	bestScore = 0
	for indices in itertools.permutations(range(len(referenceClusteringInfo))):
		matchVector = []
		for index in indices:
			match = [0] * len(referenceClusteringInfo)
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
	return mesh, saliency, clustering, numBoundaryPoints

def drawMesh(image, mesh, clustering, clusteringInfo, numBoundaryPoints):
	meshImage = image.copy()
	drawtriangles(meshImage, mesh)
	drawPoints(meshImage, mesh, clustering, clusteringInfo, numBoundaryPoints)
	return meshImage

def main():
	# sourceName = '../beach.jpg'
	sourceName = '../toss.png'
	referenceName = '../volley.png'
	# sourceName = '../cat.png'
	# referenceName = '../elephant.png'
	# sourceName = '../right tree.png'
	# sourceName = '../center tree.png'
	# sourceName = '../lego.png'

	source = cv2.imread(sourceName)
	sourceMesh, sourceSaliency, sourceClustering, sourceNumBoundaryPoints = processImage(source)
	sourceClusteringInfo = getClusterInfo(sourceMesh[sourceNumBoundaryPoints:], sourceClustering, sourceSaliency)
	sourceClustering, sourceClusteringInfo = reorderBySaliency(sourceClustering, sourceClusteringInfo)

	reference = cv2.imread(referenceName)
	referenceMesh, referenceSaliency, referenceClustering, referenceNumBoundaryPoints = processImage(reference)
	referenceClusteringInfo = getClusterInfo(referenceMesh[referenceNumBoundaryPoints:], referenceClustering, referenceSaliency)
	referenceClustering, referenceClusteringInfo = reorderBySaliency(referenceClustering, referenceClusteringInfo)

	nObjects = 3
	if len(sourceClusteringInfo) < nObjects or len(referenceClusteringInfo) < nObjects:
		print("Not enough objects found in both images")
		return

	matching = graphMatch(sourceClusteringInfo[:nObjects], source.shape, referenceClusteringInfo[:nObjects], reference.shape)
	print(matching)
	sourceClustering, sourceClusteringInfo = reorder(sourceClustering, sourceClusteringInfo, matching)

	newSourceObjectPositions, newMesh = Project.warpMesh(sourceMesh,sourceClusteringInfo, referenceClusteringInfo, sourceClustering.labels_, sourceNumBoundaryPoints, source.shape[1], source.shape[0], nObjects)

	sourceMeshImage = drawMesh(source, sourceMesh, sourceClustering, sourceClusteringInfo, sourceNumBoundaryPoints)
	referenceMeshImage = drawMesh(reference, referenceMesh, referenceClustering, referenceClusteringInfo, referenceNumBoundaryPoints)
	newMeshClustering = sourceClustering
	newMeshClusteringInfo = sourceClusteringInfo.copy()
	for i in range(len(sourceClusteringInfo)):
		# [(area,(centerX,centerY), saliencyScore),...]
		newMeshClusteringInfo[i] = (newMeshClusteringInfo[i][0], newSourceObjectPositions[i],newMeshClusteringInfo[i][2])
	newMeshImage = drawMesh(source, newMesh, newMeshClustering, sourceClusteringInfo, sourceNumBoundaryPoints)

	cv2.imshow('SourceMesh', sourceMeshImage)
	cv2.imshow('ReferenceMesh', referenceMeshImage)
	cv2.imshow('newMesh', newMeshImage)

	cv2.waitKey(0)

main()