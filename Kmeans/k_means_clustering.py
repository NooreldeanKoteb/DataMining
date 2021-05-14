# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 16:11:23 2021

@author: Nooreldean Koteb
"""

import numpy as np
from math import sqrt
import random
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

random.seed(32)


class Kmeans:
    def __init__(self, file= None, k=3, tol=0.001, iterations=-1):
        self.k = k
        self.tol = tol
        self.iterations = iterations
        self.file = file
        self.data = None

    def preprocess(self, data = None):
        if type(data) != type(None):
            return data

        with open(self.file, 'r') as f:
            lines = f.readlines()
        
        X = []
        for line in lines:
            X.append(np.array(line.split()).astype(np.float))
        
        X = np.array(X)
        
        X = normalize(X)
        
        # reducer = PCA(3)
        # X = reducer.fit_transform(X)        
        return X
        

    def euclidean_dist(self, vec1, vec2):
        return sqrt(np.sum(np.power(np.subtract(vec1, vec2), 2)))
    
    def SSE(self, vec1, vec2):
        return np.sum(np.subtract(vec1, vec2)**2)
    
    def bisect_fit(self, data=None, rand_float=True, evaluate=False):
        self.fit(cur_k=2, data=data, rand_float=rand_float)
        clusters = self.bins.copy()
        centroids = self.centroids.copy()
        centroids_SSE = self.centroid_SSE.copy()
        
        sse_val = 0.0
        for sse in centroids_SSE:
            sse_val += centroids_SSE[sse]
        
        plot_data = [[len(centroids)],[sse_val]]
        
        keys = list(centroids.keys())
                
        for i in range(self.k-2):
            largest = (0, 0)
            for clus in clusters.keys():
                size = len(clusters[clus])
                
                if size > largest[0]:
                    largest = (size, clus)
            
            cur_data = clusters[largest[1]]
            keys.remove(largest[1])
            del centroids[largest[1]]
            del clusters[largest[1]]
            del centroids_SSE[largest[1]]
            
            self.fit(cur_k=2, data=np.array(cur_data), rand_float=rand_float)
            
            key = 1
            for i in self.centroids:
                while key in keys:
                    key+=1
                
                keys.append(key)
                centroids[key] = self.centroids[i]
                clusters[key] = self.bins[i]
                centroids_SSE[key] = self.centroid_SSE[i]
            
            if len(centroids) %2 == 0:
                plot_data[0].append(len(centroids))
                
                sse_val = 0.0
                for sse in centroids_SSE:
                    sse_val += centroids_SSE[sse]
                print(sse_val)
                plot_data[1].append(sse_val)
                
                
        self.centroids = centroids
        
        if evaluate == True:
            plt.xlabel('K value')
            plt.ylabel('SSE')
            plt.plot(plot_data[0], plot_data[1])
        
        return centroids, clusters
            
    def fit(self, cur_k=None, data=None, rand_float=True):
        if cur_k == None:
            cur_k =  self.k
        #Read data and process it
        self.data = self.preprocess(data)
        
        #Get min and max of data
        self.max_data = self.data.max()
        self.min_data = self.data.min()
        self.data_size = self.data[0].shape[0]
        #Create centroids and pick random starting points from the data
        self.centroids = {}
        self.centroid_SSE = {}
        print(f'Starting Values: {self.min_data} - {self.max_data}')
        for i in range(1, cur_k+1):
            start_val = []
            for j in range(self.data_size):
#                #Improve random starting for better results
                if rand_float:
                    start_val.append(random.uniform(self.min_data, self.max_data))
                else:
                    start_val.append(random.randint(self.min_data, self.max_data))
                
            # self.centroids[i] = self.data[random.randint(0, len(self.data)-1)]
            
            self.centroids[i] = start_val
            self.centroid_SSE[i] = 0.0
            # print(f'Centroid-{i}: {start_val}')
       
        iters = 0
        while True:
            iters+=1
            #Checks if any of the centroids are larger than the given tolerance
            check_tol = True
            for i in self.centroids.keys():
                SSE_val = 0.0

                for j in self.data:
                    SSE_val += self.SSE(j,self.centroids[i])
                
                change = np.abs(self.centroid_SSE[i]-SSE_val)
                print(f"Iter-{iters}: Centroid-{i} Change-{change}")
                if change > self.tol:
                    check_tol =  False
                
                self.centroid_SSE[i] = SSE_val
            print('\n')
            
            if check_tol == True:
                break
            
            #classify data based on distance to centroids into bins
            self.bins = {}
            for i in range(1, cur_k+1):
                self.bins[i] = []
            
            #Measures distances between data and centroids
            for i in self.data:
                dist = []
                for j in self.centroids.keys():
                    dist.append((self.SSE(i, self.centroids[j]), j))
                
                self.bins[min(dist)[1]].append(i)


            #recalculates centroids
            for i in self.bins:
                bin_avg = np.average(self.bins[i], axis=0)
                if np.isnan(np.sum(bin_avg)):
#                    #If bin is empty find better random starting points
                    bin_avg = []
                    for j in range(self.data_size):
                        if rand_float:
                            bin_avg.append(random.uniform(self.min_data, self.max_data))
                        else:
                            bin_avg.append(random.randint(self.min_data, self.max_data))
                            
                self.centroids[i] = bin_avg
                    # self.centroids[i] = self.data[random.randint(0, len(self.data)-1)]
                
            # #Checks if max iterations were reached, or if iterations set to -1
            # if self.iterations == -1:
            #     continue
            # elif self.iterations == iters:
            #     print(f"Maximum iterations reached: {self.iterations}")
            #     break
        
        
    def predict(self, data= None):
        self.data = self.preprocess(data)
                
        self.results = []
        
        for i in self.data:
            dist = []
            for j in self.centroids.keys():
                dist.append((self.SSE(i, self.centroids[j]), j))
            
            self.results.append(min(dist)[1])
        
        return self.results

    def write(self, out_file):
        with open(out_file,"w") as f:
            for i in self.results:
                f.write(f'{i}\n')  
                
        f.close()


model = Kmeans('iris_new_data.txt', k=3, tol=0.001, iterations=-1)
# model.fit()
model.bisect_fit()
results = model.predict()
model.write('Iris_Result.txt')


