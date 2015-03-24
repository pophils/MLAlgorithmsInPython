
import numpy as NP 
from matplotlib import pyplot as PP
class KMeans(object):
    """
        Class provides implementation of the basic 
        KMeans unsupervised machine Learning problems.
        It produces K clusters of similar items from a 
        given datasets. Each cluster is described by its 
        centre known as the centroid.           
        Its easy to implement but with a complexity that grows k

        Pseudocode:
            Create K Centroid points from a random distribution
               Repeat while any centroid point has changed
                 For every point in the data set 
                   for every centroid point
                     calculates the distance between the centroid and data point
                   assign point to cluster with the least distance
                For every new clusters of point
                 find the mean point and update the centroid with point.
   """

    def __init__(self, datasets, k):
      
       if len(datasets) < 1:
           raise ValueError("The datasets supplied is empty.")

       if k < 1:
           raise ValueError("K supplied is less than 1.")

       #if isinstance(datasets, NP.ndarray) == False:
       #    self.datasets = NP.array(datasets)
       #else:
       #    self.datasets = datasets

       self.k = k
       self.datasets = NP.mat(datasets)

    def GetEuclideanDistance(self, vectA, vectB):
        # vectA = NP.array(vectA)   
         #vectB = NP.array(vectB)             
         
         return NP.sqrt(NP.sum(NP.power((vectA - vectB), 2)))

    def GetDistance(self, vectA, vectB, distanceFunction = GetEuclideanDistance):
        return distanceFunction(self, vectA, vectB)

    def GenerateKCentroids(self):
        self.featureDimension = self.datasets.shape[1]
        self.noOfDatasets = self.datasets.shape[0]
        self.centroids = NP.mat(NP.zeros((self.k, self.featureDimension)))

        for column in range(self.featureDimension):
            #assume random number gen from 0 - 1 as a normalized value
            # so convert the normalized value to the weight of each dataset column

            minFeatureVal = NP.min(self.datasets[:, column])
            rangeDenominator = NP.float(NP.max(self.datasets[:, column]) - minFeatureVal)
            centroid = minFeatureVal + rangeDenominator * NP.random.rand(self.k, 1)
            self.centroids[:, column] = centroid
        return self.centroids

    def InitKCentroids(self, func = GenerateKCentroids):
        func(self)

    def Cluster(self):
        
        self.InitKCentroids()
        centroidChanged = True
        self.pointCentroidMap = NP.mat(NP.zeros((self.noOfDatasets,2)))

        while centroidChanged:
            centroidChanged = False
            
            for datasetRow in range(self.noOfDatasets):
                minDistance = NP.inf
                minDistanceK = None

                for centroidRow in range(self.k):
                    distance = self.GetEuclideanDistance(self.datasets[datasetRow, :], self.centroids[centroidRow, :])
                    if distance < minDistance:
                        minDistance = distance
                        minDistanceK = centroidRow

                if self.pointCentroidMap[datasetRow,0] != minDistanceK:
                    centroidChanged = True

                self.pointCentroidMap[datasetRow, :] = minDistanceK, minDistance ** 2 # square distance

            for centroidIndex in range(self.k):
                #cluster = [self.datasets[row, :] for row in range(self.noOfDatasets) if self.pointCentroidMap.get(row)[0] == centroidIndex]
               
                cluster = self.datasets[NP.nonzero(self.pointCentroidMap[:, 0].A == centroidIndex)[0]]
                self.centroids[centroidIndex, :] = NP.mean(cluster, axis=0)                 

        return self.pointCentroidMap, self.centroids

    def PlotCluster(self):
        fig = PP.figure()
        #fig.subplots_adjust(left=1, bottom=0, top=0, right=1, wspace=0.05, hspace=0.05)


        datasetSubPlot = fig.add_subplot(121)
        clusterSubPlot = fig.add_subplot(122)
                
        datasetSubPlot.scatter(self.datasets[:, 0], self.datasets[:, 1], label='Original Plot of The Datasets') 

        colorSet = [0.4,1,0.4], [1, 0.4, 0.4], [0.1,0.8,1]
        colors = [colorSet[NP.int(self.pointCentroidMap[point,0])] for point in range(self.noOfDatasets)]

        clusterSubPlot.scatter(self.datasets[:, 0], self.datasets[:, 1], c=colors, label='Cluster plot of the datasets')

        clusterSubPlot.scatter(self.centroids[:, 0], self.centroids[:, 1], color='yellow',  marker='x')

        fig.show()                  
 
    def kMeans(self):
        m = NP.shape(self.datasets)[0]
        self.clusterAssment = NP.mat(NP.zeros((m,2))) 
        centroids = self.GenerateKCentroids()
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = NP.inf; minIndex = -1
                for j in range(self.k):                                
                    distJI = self.GetEuclideanDistance(centroids[j,:],self.datasets[i,:])  
                    if distJI < minDist:                          
                        minDist = distJI; minIndex = j
                if self.clusterAssment[i,0] != minIndex: 
                    clusterChanged = True
                self.clusterAssment[i,:] = minIndex,minDist**2
            for cent in range(self.k):  
               ptsInClust = self.datasets[NP.nonzero(self.clusterAssment[:,0].A==cent)[0]]
               centroids[cent,:] = NP.mean(ptsInClust, axis=0)                 
               
        return centroids, self.clusterAssment

    def PlotCluster2(self):
        fig = PP.figure()
        #fig.subplots_adjust(left=1, bottom=0, top=0, right=1, wspace=0.05, hspace=0.05)


        datasetSubPlot = fig.add_subplot(121)
        clusterSubPlot = fig.add_subplot(122)
                
        datasetSubPlot.scatter(self.datasets[:, 0], self.datasets[:, 1], label='Original Plot of The Datasets') 

        colorSet = [0.4,1,0.4], [1, 0.4, 0.4], [0.1,0.8,1]
        colors = [colorSet[NP.int(self.clusterAssment[point,0])] for point in range(self.noOfDatasets)]

        clusterSubPlot.scatter(self.datasets[:, 0], self.datasets[:, 1], c=colors, label='Cluster plot of the datasets')

        fig.show()      
 

   
