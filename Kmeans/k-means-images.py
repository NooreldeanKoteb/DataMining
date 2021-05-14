# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:58:29 2021

@author: Nooreldean Koteb
"""

#My code from part 1
import k_means_clustering as km
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, FactorAnalysis
import numpy as np

from sklearn.manifold import TSNE

def preprocess(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    
    X = []
    for line in lines:
        X.append(np.array(line.split(',')).astype(np.float))
    
    X = np.array(X)


    remove = []
    for col in range(len(X[0])):
        if np.sum(X[:, col]) == 0:
            remove.append(col)
    

    new_X = []
    for row in range(len(X)):
        temp = []
        for col in range(len(X[row])):
            if col not in remove:
                if X[row][col] == 0:
                    temp.append(1)
                else:
                    temp.append(0)#Highest score with this
                # temp.append(X[row][col])
                
                    
        new_X.append(temp)
    X = np.array(new_X)
    
    # reducer = PCA(200)
    # X = reducer.fit_transform(X)


    X = TSNE(n_jobs=-1).fit_transform(X) #50
    

    
    # X = normalize(X) #l1
        
    # reducer = PCA(0.95)
    # X = reducer.fit_transform(X)
    
    #Figure out how to lower data to improve results
    

    # reducer = FactorAnalysis(600)
    # X = reducer.fit_transform(X)
    
    return X
X = preprocess('new_test4.txt')


model = km.Kmeans(k=10, tol=0.001, iterations=-1)

# # model.fit(preprocess=False, data=X, rand_float=True)
model.bisect_fit(data=X, rand_float=True, evaluate=True)
y = model.predict(data=X)
model.write('Images_Result.txt')
