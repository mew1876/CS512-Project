import numpy as np
cimport numpy as np
cimport cython

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)

# todo: add type for Info parameters
# todo: add imageHeight for check on possibleQnSet creation
def warpMesh(long[:,:] mesh, sourceInfo, referenceInfo, long long[:] sourceClusteringLabels, long long numBoundaryPoints, long long imageWidth, long long imageHeight, long long nObjects): #clusterInfo = [(area,(centerX,centerY), saliencyScore),...]
	cdef int radiusMax = int(.05*imageWidth)
	cdef float alpha = 0.1
	cdef float beta = 0.1
	cdef float gamma = 0.1

	cdef int lenSourceInfo = len(sourceInfo)
	cdef int lenMesh = len(mesh)

	cdef int u, n, row, col, qIndex, k, cluster, pointIndex # for loop variables

	# sort referenceInfo by object size, and sort sourceInfo the same way
	referenceInfoSorted = sorted(referenceInfo,key=lambda info: info[0])
	sourceInfoSorted = []
	for sortedIndex in range(lenSourceInfo): # index after object size sort in beginning
		unsortedIndex = referenceInfo.index(referenceInfoSorted[sortedIndex]) # index it should return as
		# looping through the objects in sorted order, so appending here will sort properly
		sourceInfoSorted.append(sourceInfo[unsortedIndex])

	# variables defined in the loops in the order they appear
	cdef np.ndarray[np.float32_t, ndim=2] calculation = np.zeros((nObjects,lenMesh), dtype=np.float32)
	# cdef np.ndarray[np.uint8_t,ndim=2] finalQPositions
	cdef np.ndarray[np.float32_t,ndim=1] objectiveD
	cdef np.ndarray[np.float32_t,ndim=1] objectiveE
	# cdef int[][] possibleQnSet
	cdef np.ndarray[np.int_t, ndim=3] newMesh = np.zeros((nObjects,lenMesh,2), dtype=np.int_)
	cdef float[2] qn = [<float>0.0,<float>0.0]
	cdef float[2] sn = [<float>0.0,<float>0.0]
	cdef float[2] pn = [<float>0.0,<float>0.0]
	cdef float currentradius
	cdef float[2] newMeshCalcPt1 = [<float>0.0,<float>0.0]
	cdef float[2] newMeshCalcPt2 = [<float>0.0,<float>0.0]
	cdef float newMeshCalcPt3
	cdef float objectiveDPt1
	cdef float objectiveDPt2
	cdef float objectiveEPt1
	cdef float objectiveEPt2Outer
	cdef int label
	cdef float objectiveEPt2Inner
	cdef np.ndarray[np.float32_t,ndim=1] objectives
	cdef int minIndex

	for u in range(lenMesh): # for each point in the mesh
		for n in range(nObjects): # for each object number
			calculation[n,u] = 1/(np.linalg.norm(mesh[u]-sourceInfoSorted[n][1])**2)
	# loop 1
	finalQPositions = [] # final position decisions!
	for n in range(nObjects): 
		print "calculating optimal location for center of cluster #"+str(n)
		# objective functions for possibilities
		objectiveD = np.zeros((2*radiusMax)**2, dtype=np.float32)
		objectiveE = np.zeros((2*radiusMax)**2, dtype=np.float32)
		# possible qn values
		possibleQnSet = []

		# calculation variables
		sn[0] = <float>sourceInfoSorted[n][1][0]
		sn[1] = <float>sourceInfoSorted[n][1][1]
		pn[0] = <float>referenceInfoSorted[n][1][0]
		pn[1] = <float>referenceInfoSorted[n][1][1]

		# loop 2 setup
		for row in range(-radiusMax,radiusMax): 
			for col in range(-radiusMax,radiusMax):
				if pn[1]+col >= 0 and pn[1]+col < imageWidth:
					if pn[0]+row >= 0 and pn[0]+row < imageHeight:
						possibleQnSet.append([pn[0]+row,pn[1]+col])
		# loop 2
		# mesh calculation
		# newMesh = np.zeros(((2*radiusMax)**2,lenMesh,2), dtype=np.int_)
		for qIndex in range(len(possibleQnSet)):
			qn[0] = possibleQnSet[qIndex][0]
			qn[1] = possibleQnSet[qIndex][1]
			currentRadius = np.absolute(np.linalg.norm(qn-pn))
			if currentRadius <= radiusMax: # check distance constraint
				# loop 3
				for u in range(lenMesh): 
					newMeshCalcPt1 = [<float>(calculation[n,u]*(mesh[u][0] + qn[0] - sn[0])),<float>(calculation[n,u]*(mesh[u][1] + qn[1] - sn[1]))]
					newMeshCalcPt2 = [<float>0.0,<float>0.0]
					newMeshCalcPt3 = 0
					# calculate Pt2 and Pt3's summations
					for k in range(n+1):
						if not k == n: # don't add to the summation when k==n for Pt2
							newMeshCalcPt2[0] += calculation[k,u]*newMesh[k,u][0]
							newMeshCalcPt2[1] += calculation[k,u]*newMesh[k,u][1]
						newMeshCalcPt3 += calculation[k,u]
					newMesh[n,u] = [(newMeshCalcPt1[0]+newMeshCalcPt2[0])/newMeshCalcPt3,(newMeshCalcPt1[1]+newMeshCalcPt2[1])/newMeshCalcPt3]
			else: # can't leave untouched numbers as zero because we need the minimum!
				objectiveD[qIndex] = 99999999
				objectiveE[qIndex] = 99999999
				continue
				# end loop 3
			
			# calculate objective D function
			# objectiveDPt1 = 1
			# for n in range(nObjects):
			objectiveDPt1 = (1+alpha*(currentRadius)) 
			objectiveDPt2 = 1 # we didn't account for boundary cropping, so this is just 1
			objectiveD[qIndex] = objectiveDPt1 * objectiveDPt2

			# calculate objective E function
			proportionFlipOverTriangles = 0 # todo: calculate proportion of flip-over triangles for newMesh[qIndex]
			objectiveEPt1 = 1 + (gamma*proportionFlipOverTriangles)
			objectiveEPt2Outer = 0
			# for cluster in range(lenSourceInfo): # loop over the clusters, not the amt of objects
			# 	objectiveEPt2Inner = 0
			# 	for pointIndex in range(len(sourceClusteringLabels)):
			# 		label = sourceClusteringLabels[pointIndex]
			# 		if label == cluster:
			# 			objectiveEPt2Inner += (np.linalg.norm(np.subtract((newMesh[n,numBoundaryPoints:])[pointIndex], (mesh[numBoundaryPoints:])[pointIndex])))**2
			# 	objectiveEPt2Outer += objectiveEPt2Inner
			# objectiveE[qIndex] = objectiveEPt1*objectiveEPt2Outer
			objectiveEPt2Inner = 0
			for pointIndex in range(len(sourceClusteringLabels)):
				label = sourceClusteringLabels[pointIndex]
				if label != -1:
					objectiveEPt2Inner += (np.linalg.norm(np.subtract((newMesh[n,numBoundaryPoints:])[pointIndex], (mesh[numBoundaryPoints:])[pointIndex])))**2
			objectiveEPt2Outer += objectiveEPt2Inner
			objectiveE[qIndex] = objectiveEPt1*objectiveEPt2Outer
			# end calculating objective functions
		# end loop 2
		# pick qn based on objective functions
		objectives = objectiveD*objectiveE
		minIndex = np.argmin(objectives)
		finalQPositions.append(possibleQnSet[minIndex])
	# end loop 1

	# return objects to original cluster ordering
	finalQPositionsUnsorted = []
	for unsortedIndex in range(lenSourceInfo): # index after object size sort in beginning
		sortedIndex = sourceInfoSorted.index(sourceInfo[unsortedIndex]) # index it should return as
		# looping through the objects in original/unsorted order, so appending here will sort properly
		finalQPositionsUnsorted.append(finalQPositions[sortedIndex])
	return finalQPositionsUnsorted, newMesh[nObjects-1]
