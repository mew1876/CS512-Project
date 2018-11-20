# Michael Walsh & Julianna Bochnak

import random
import itertools
from time import sleep
import sys
import os
import re
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
	for row in range(edgeSuppressDistance * 2, rows - edgeSuppressDistance * 2):
		for col in range(edgeSuppressDistance * 2, cols - edgeSuppressDistance * 2):
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

def getTriangles(mesh, imageShape):
	rect = (0, 0, imageShape[0], imageShape[1])
	subdiv = cv2.Subdiv2D(rect);
	subdiv.insert(list(mesh))
	triangles = subdiv.getTriangleList();
	filteredIndexTriangles = []
	for triangle in triangles:
		if (rect[0] <= triangle[0] < rect[2]
			and rect[0] <= triangle[2] < rect[2]
			and rect[0] <= triangle[4] < rect[2]
			and rect[1] <= triangle[1] < rect[3]
			and rect[1] <= triangle[3] < rect[3]
			and rect[1] <= triangle[5] < rect[3]):
				cornerIndices = []
				for index in range(0, 6, 2):
					cornerIndices.append(np.where((mesh == [triangle[index], triangle[index + 1]]).all(axis=1))[0][0])
				filteredIndexTriangles.append(cornerIndices)
	filteredIndexTriangles = np.array(filteredIndexTriangles)
	return filteredIndexTriangles

def drawTriangles(image, mesh, triangles):
	for triangle in triangles:
		cv2.line(image, (mesh[triangle[0], 1], mesh[triangle[0], 0]), (mesh[triangle[1], 1], mesh[triangle[1], 0]), (0,0,0), 1, cv2.LINE_8, 0)
		cv2.line(image, (mesh[triangle[0], 1], mesh[triangle[0], 0]), (mesh[triangle[2], 1], mesh[triangle[2], 0]), (0,0,0), 1, cv2.LINE_8, 0)
		cv2.line(image, (mesh[triangle[1], 1], mesh[triangle[1], 0]), (mesh[triangle[2], 1], mesh[triangle[2], 0]), (0,0,0), 1, cv2.LINE_8, 0)

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
	_, centers, _ = zip(*clusteringInfo)
	for index, center in enumerate(centers):
		cv2.circle(image, (int(center[1]), int(center[0])), 20, colors[index + 1], thickness=cv2.FILLED)

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
		minRow = -1
		maxRow = 0
		minCol = -1
		maxCol = 0
		for point in cluster:
			if point[0] < minRow or minRow == -1:
				minRow = point[0]
			elif point[0] > maxRow:
				maxRow = point[0]
			if point[1] < minCol or minCol == -1:
				minCol = point[1]
			elif point[1] > maxCol:
				maxCol = point[1]
		centers.append(np.array([minRow + (maxRow - minRow) / 2, minCol + (maxCol - minCol) / 2]))
		# centers.append(sum(contour) / len(contour))
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
	reordering = list(enumerate(saliencyScores))
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

def	transformImage(image, originalMesh, transformedMesh, triangles):
	transformedImage = np.zeros(image.shape, dtype=np.uint8)
	for triangle in triangles:
		triangleContour = np.array([transformedMesh[triangle[0]], transformedMesh[triangle[1]], transformedMesh[triangle[2]]])
		signedArea = cv2.contourArea(triangleContour)
		if signedArea > 0:
			# Triangle isn't flipped
			minRow = min(transformedMesh[triangle[0]][0], transformedMesh[triangle[1]][0], transformedMesh[triangle[2]][0])
			maxRow = max(transformedMesh[triangle[0]][0], transformedMesh[triangle[1]][0], transformedMesh[triangle[2]][0])
			minCol = min(transformedMesh[triangle[0]][1], transformedMesh[triangle[1]][1], transformedMesh[triangle[2]][1])
			maxCol = max(transformedMesh[triangle[0]][1], transformedMesh[triangle[1]][1], transformedMesh[triangle[2]][1])
			for row in range(minRow, maxRow):
				for col in range(minCol, maxCol):
					if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
						if cv2.pointPolygonTest(triangleContour, (row, col), False) >= 0:
							# Point is in the triangle
							barry01 = cv2.contourArea(np.array([[row, col], transformedMesh[triangle[0]], transformedMesh[triangle[1]]])) / signedArea
							barry02 = cv2.contourArea(np.array([[row, col], transformedMesh[triangle[0]], transformedMesh[triangle[2]]])) / signedArea
							barry12 = 1 - barry01 - barry02
							floatCoords = barry12 * originalMesh[triangle[0]] + barry02 * originalMesh[triangle[1]] + barry01 * originalMesh[triangle[2]]
							if 0 <= int(floatCoords[0]) < image.shape[0] and 0 <= int(floatCoords[1]) < image.shape[1]:
								transformedImage[row, col] = image[int(floatCoords[0]), int(floatCoords[1])]
	return transformedImage

def processImage(image):
	canny = edgeDetection(image)
	saliency = getSaliency(image)
	_,  processedSaliency = cv2.threshold(saliency, 30, 255, cv2.THRESH_BINARY)
	mesh, numBoundaryPoints = createMesh(canny, processedSaliency)
	meshWithoutBoundaries = np.array(mesh[numBoundaryPoints:])
	clustering = clusterPoints(meshWithoutBoundaries)
	return mesh, saliency, clustering, numBoundaryPoints

def drawMesh(image, mesh, triangles, clustering, clusteringInfo, numBoundaryPoints):
	meshImage = image.copy()
	drawTriangles(meshImage, mesh, triangles)
	drawPoints(meshImage, mesh, clustering, clusteringInfo, numBoundaryPoints)
	return meshImage

def main():
	# sourceName = '../beach.jpg'
	sourceName = '../toss.png'
	referenceName = '../volley.png'
	# referenceName = '../volleyResize.png'
	# sourceName = '../cat.png'
	# referenceName = '../elephant.png'
	# sourceName = '../right tree.png'
	# sourceName = '../center tree.png'
	# sourceName = '../lego.png'

	nObjects = 3
	random.seed(1)

	if len(sys.argv) == 4:
		sourceName = sys.argv[1]
		referenceName = sys.argv[2]
		nObjects = int(sys.argv[3])
	else:
		print("Usage: python R2P.py <source> <reference> <number of objects>")
		return

	folder = re.match(r'.*/([^/]+)\.', sourceName).group(1) + '/'
	if not os.path.exists(folder):
	    os.makedirs(folder)

	print("Recomposing", sourceName, "to match", referenceName)
	print("Getting source mesh")
	source = cv2.imread(sourceName)
	sourceMesh, sourceSaliency, sourceClustering, sourceNumBoundaryPoints = processImage(source)
	cv2.imwrite(folder + 'sourceSaliency.png', sourceSaliency)
	cv2.imshow('SourceSaliency', sourceSaliency)
	sourceClusteringInfo = getClusterInfo(sourceMesh[sourceNumBoundaryPoints:], sourceClustering, sourceSaliency)
	sourceClustering, sourceClusteringInfo = reorderBySaliency(sourceClustering, sourceClusteringInfo)
	sourceTriangles = getTriangles(sourceMesh, source.shape)
	sourceMeshImage = drawMesh(source, sourceMesh, sourceTriangles, sourceClustering, sourceClusteringInfo, sourceNumBoundaryPoints)
	cv2.imshow('SourceMesh', sourceMeshImage)

	print("Getting reference mesh")
	reference = cv2.imread(referenceName)
	referenceMesh, referenceSaliency, referenceClustering, referenceNumBoundaryPoints = processImage(reference)
	cv2.imwrite(folder + 'referenceSaliency.png', referenceSaliency)
	cv2.imshow('ReferenceSaliency', referenceSaliency)
	referenceClusteringInfo = getClusterInfo(referenceMesh[referenceNumBoundaryPoints:], referenceClustering, referenceSaliency)
	referenceClustering, referenceClusteringInfo = reorderBySaliency(referenceClustering, referenceClusteringInfo)
	referenceTriangles = getTriangles(referenceMesh, reference.shape)
	referenceMeshImage = drawMesh(reference, referenceMesh, referenceTriangles, referenceClustering, referenceClusteringInfo, referenceNumBoundaryPoints)
	cv2.imshow('ReferenceMesh', referenceMeshImage)

	cv2.waitKey(1)

	if len(sourceClusteringInfo) >= nObjects and len(referenceClusteringInfo) >= nObjects:
		print("Matching objects")
		matching = graphMatch(sourceClusteringInfo[:nObjects], source.shape, referenceClusteringInfo[:nObjects], reference.shape)
		sourceClustering, sourceClusteringInfo = reorder(sourceClustering, sourceClusteringInfo, matching)
		print(matching)
		sourceMeshImage = drawMesh(source, sourceMesh, sourceTriangles, sourceClustering, sourceClusteringInfo, sourceNumBoundaryPoints)
		cv2.imshow('SourceMesh', sourceMeshImage)
		cv2.waitKey(1)

		print("Warping mesh")
		newSourceObjectPositions, newMesh = Project.warpMesh(sourceMesh, sourceClusteringInfo, referenceClusteringInfo, sourceClustering.labels_, sourceNumBoundaryPoints, source.shape[1], source.shape[0], nObjects)
		print("Warping image")
		transformedSource = transformImage(source, sourceMesh, newMesh, sourceTriangles)
		newMeshClustering = sourceClustering
		newMeshClusteringInfo = sourceClusteringInfo.copy()
		for i in range(len(sourceClusteringInfo)):
			# [(area,(centerX,centerY), saliencyScore),...]
			newMeshClusteringInfo[i] = (newMeshClusteringInfo[i][0], newSourceObjectPositions[i],newMeshClusteringInfo[i][2])
		newMeshImage = drawMesh(source, newMesh, sourceTriangles, newMeshClustering, sourceClusteringInfo, sourceNumBoundaryPoints)
		cv2.imshow('NewMesh', newMeshImage)
		cv2.imshow('Recomposed', transformedSource)
		cv2.imwrite(folder + 'newMesh.png', newMeshImage)
		cv2.imwrite(folder + 'recomposed.png', transformedSource)
	else:
		print("Not enough objects found in both images")

	cv2.imwrite(folder + 'sourceMesh.png', sourceMeshImage)
	cv2.imwrite(folder + 'referenceMesh.png', referenceMeshImage)

	cv2.waitKey(0)

main()