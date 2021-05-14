# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:38:31 2021

@author: Nooreldean Koteb
"""
# #Hide useless SKlearn warnings
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn


import json
import scipy
import torch
import json
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from itertools import product
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
#, precision_recall_fscore_support



import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable



class NN(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_out, dropout=None):
        super(NN, self).__init__()
        print(f'Creating a Neural Network with {n_layer+1} layer(s)')
        print(f'Input: {n_input} - Hidden: {n_hidden} - Output: {n_out} - Dropout: {dropout}')
        
        self.layers = n_layer
        
        self.fc1 = nn.Linear(n_input, n_hidden)

        if n_layer > 1:
            for i in range(2, n_layer+1):
                exec(f'self.fc{i} = nn.Linear(n_hidden, n_hidden)')

        exec(f'self.fc{n_layer+1} = nn.Linear(n_hidden, n_out)')

        if dropout != None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        if self.layers > 1:
            for i in range(2, self.layers+1):
                x = self.dropout(x)
                x = F.relu(exec(f'self.fc{i}(x)'))
                    
        x = self.dropout(x)
        x = F.softmax(exec(f'self.fc{self.layers+1}(x)'))


def use_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype=None
    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        print()
    return (device, dtype)

def preprocess(device, dtype, train_dat=None, pred_dat=None, sc=None, reducer=None,
               reduction=None, resample_val=None, cuda_nn=False):
    X_data = None
    X_pred = None
    y_data = None
    try:
        print('Attempting to read Data...')
        if train_dat != None:
            dataset = pd.read_csv(train_dat, header=None)
            if resample_val == None:
                X_data = dataset.iloc[:, :-1 ].values
                y_data = dataset.iloc[:, -1].values
    
            #Resampling
            else:
                print('Resampling Data...')
                df = pd.DataFrame(dataset.values)
                df_bins = {}
                for i in range(1,8):
                    val = df.loc[df[54]==i]
                    df_bins[val.shape[0]] = val
                
                if resample_val[0] == 'max':
                    n_samples = max(df_bins.keys())
                elif resample_val[0] == 'min':
                    n_samples = min(df_bins.keys())
                else:
                    n_samples = resample_val[0]
                replace = resample_val[1]
                stratify=None
                # if resample_val[2] == True:
                #     stratify = (X_data, y_data)
                
                # data = []
                X_data = []
                y_data = []
                for i in df_bins.keys():
                    vals = resample(df_bins[i].to_numpy(), replace=replace, n_samples=n_samples, stratify=stratify)
                    X_data.extend(vals[:, :-1].tolist())
                    y_data.extend(vals[:, -1].tolist())
            
                X_data = np.array(X_data)
                y_data = np.array(y_data)
                
            
            #Scaling data
            sc = StandardScaler()
            X_data = sc.fit_transform(X_data)
    
            #Data Reduction
            if reduction != None:
                print('Data reduction commencing...')
                if reduction[0] == 'PCA':
                    reducer = PCA(reduction[1])
                    X_data = reducer.fit_transform(X_data)
    
                elif reduction[0] == 'FA':
                    reducer = FactorAnalysis(reduction[1])
                    X_data = reducer.fit_transform(X_data)
                    
            if device.type == 'cuda' and cuda_nn == True:
                X_data = torch.tensor(X_data, device=device)
                y_data = torch.tensor(y_data, device=device)
    
        if pred_dat != None:
            dataset = pd.read_csv(pred_dat, header=None)
            X_pred = dataset.values
    
            if sc != None:
               X_pred = sc.transform(X_pred)
            else:
                print('Please give a scaler for your test data...')
                return
    
            if reducer != None:
                X_pred = reducer.transform(X_pred)
    
            if device.type == 'cuda'  and cuda_nn == True:
                X_pred = torch.tensor(X_pred, device=device)
    except:
        print('Failed to load data!')


    return (X_data, y_data, X_pred, sc, reducer)



def ANN_train(X, y, model, optimizer_lr=('SGD', 0.01), epochs=20, patience=None):
    optimizer = None
    if optimizer[0] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=optimizer[1])
    elif optimizer[1] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=optimizer[1])

    loss_function = nn.CrossEntropyLoss()
    
    last_model = None
    for epoch in range(epochs):
        if epoch != 0:
                print('\repoch: '+str(epoch)+' - 100.00%', end='', flush=True)
        #Early stopping
        if patience != None and epoch%patience == 0:
            
            
            
            acc = accuracy_score(scores, y)
            if epoch == 0:
                print(f'\nEarly stopping Accuracy before training: {acc}')
            else:
                print(f'\nEarly stopping Accuracy for epoch {epoch}: {acc}')
                
            if acc > last_model[0]:
                last_model = (acc, model)
            else:
                print('No improvement in Accuracy, taking the last best model and ending training.')
                model = last_model[1]
                break
            
        print(f"\nStarting epoch {epoch+1}...")
        progress = 0
        t = dt.datetime.now()
        num_data = len(X)
        for forest in X:
            progress +=1
            if (dt.datetime.now() -t).seconds / 10 >= 1:
                t = dt.datetime.now()
                percent = progress/num_data
                print('\repoch: '+str(epoch+1)+' - {:.2%}'.format(percent), end='', flush=True)        
        
            scores = model(X)
            
            loss = loss_function(scores, y)
            loss.backward()
    
            optimizer.step()
        
    return model

def ANN(n_input, n_hidden, n_layer, n_out, device, dropout=None):
    model = NN(n_input, n_hidden, n_layer, n_out, dropout=dropout)
    
    if device.type == 'cuda':
        model.cuda()
    
    return model

def KNN(X, y, k, metric):
    print('Creating KNN classifer!')
    clf = KNeighborsClassifier(n_neighbors = k, metric=metric, n_jobs=-1)
    print('Fitting KNN Classifer!')
    clf.fit(X,y)
    return clf

def DT(X, y, func, depths=None, features=None, leafs=None):
    print('Creating Decision Tree classifer!')
    clf = DecisionTreeClassifier(criterion=func, max_depth=depths,
                                max_features=features, random_state=1,
                                max_leaf_nodes=leafs)
    print('Fitting Decision Tree Classifer!')
    clf.fit(X, y)
    return clf

def RF(X, y, n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf,
       min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, 
       ccp_alpha, max_samples, n_jobs=-1):
    print('Creating Random Forest classifer!')
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion= criterion, 
                                 max_depth=max_depth, min_samples_split=min_samples_split, 
                                 min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, 
                                 max_features=max_features, max_leaf_nodes=max_leaf_nodes, 
                                 min_impurity_decrease=min_impurity_decrease, n_jobs=-1, ccp_alpha=ccp_alpha, max_samples=max_samples)
    
    print('Fitting Random Forest  Classifer!')
    clf.fit(X, y)
    return clf

def NB(X, y):
    print('Creating Naive Bayes classifer!')
    clf = GaussianNB()
    print('Fitting Naive Bayes Classifer!')
    clf.fit(X,y)
    return clf

def evaluation(test, pred):
    print('\nEvaluation - Confusion Matrix:')
    cm = confusion_matrix(test, pred)
    print(cm)
    f1 = f1_score(test, pred, average='weighted')
    print('F1-Score: '+str(f1))
    acc = accuracy_score(test, pred)
    print('Accuracy: '+str(acc))
    # prf = precision_recall_fscore_support(test, pred)
    # print(prf)
    return {'confusion_matrix': cm,
            'f1_score': f1,
            'accuracy_score': acc,}



def cross_val(train_dat, clf_type, parameters, device, dtype, cuda_nn=False):
    keys, values = zip(*parameters.items())
    params = []
    for val in product(*values):
        params.append(dict(zip(keys, val)))

    results = {}
    X_train = []
    cur_clf = []
    cur_data = []
    clf = None
    for param in params:
        print('\n'+str(param))
        
        reduction = param['reduction']
        resample_val = param['resample_val']
        
        #speed some preprocessing time
        if cur_data == [] or cur_data != [reduction, resample_val]:
            cur_data = [reduction, resample_val]

            X_data, y_data, _, _, _ = preprocess(device, dtype, train_dat=train_dat, 
                                                 pred_dat=None, sc=None, reducer=None,
                                                 reduction=reduction, resample_val=resample_val,
                                                 cuda_nn = cuda_nn)
            
            #split data
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, shuffle=True)
        
        
        if clf_type == 'KNN':
            clf = KNN(X_train, y_train, param['k'], param['metric'])
                
        elif clf_type == 'DT':
            clf = DT(X_train, y_train, param['func'], param['depths'], 
                      param['features'], param['leafs'])
               
        elif clf_type == 'NB':
            clf = NB(X_train, y_train)

        elif clf_type == 'RF':
            clf = RF(X_train, y_train, param['n_estimators'], param['criterion'], param['max_depth'], 
                     param['min_samples_split'], param['min_samples_leaf'], 
                     param['min_weight_fraction_leaf'], param['max_features'], param['max_leaf_nodes'], 
                     param['min_impurity_decrease'], param['ccp_alpha'], param['max_samples']) 
            
        elif clf == 'ANN':
            # clf = ANN(param['n_input'], param['n_hidden'], param['n_layer'], 
            #           param['n_out'], device, param['dropout'])
            
            # clf = ANN_train(X_train, y_train, clf, param['optimizer_lr'], 
            #                 param['epochs'], param['patience'])
            pass
        else:
            return print('Please designate a classifier!')


        y_pred = []
        print('Starting Test...')
        y_pred = clf.predict(X_test)
        scores = evaluation(y_test, y_pred)

        if scores['accuracy_score'] in results:
            results[scores['accuracy_score']].append([param, scores])
        else:
            results[scores['accuracy_score']] = [[param, scores]]

    return results




def final_predict(train_dat, test_dat, clf_type, param, device, dtype, cuda_nn=False):
    reduction = param['reduction']
    resample_val = param['resample_val']
    
    #speed some preprocessing time
    X_data, y_data, X_pred, _, _ = preprocess(device, dtype, train_dat=train_dat, 
                                             pred_dat=test_dat, sc=None, reducer=None,
                                             reduction=reduction, resample_val=resample_val,
                                             cuda_nn = cuda_nn)
    
    if clf_type == 'KNN':
        clf = KNN(X_data, y_data, param['k'], param['metric'])
            
    elif clf_type == 'DT':
        clf = DT(X_data, y_data, param['func'], param['depths'], 
                  param['features'], param['leafs'])
            
    elif clf_type == 'NB':
        clf = NB(X_data, y_data)
        
    elif clf_type == 'RF':
        clf = RF(X_data, y_data, param['n_estimators'], param['criterion'], param['max_depth'], 
                 param['min_samples_split'], param['min_samples_leaf'], 
                 param['min_weight_fraction_leaf'], param['max_features'], param['max_leaf_nodes'], 
                 param['min_impurity_decrease'], param['ccp_alpha'], param['max_samples']) 
    elif clf == 'ANN':
        #Ill get back to you
        pass
    else:
        return print('Please designate a classifier!')


    y_pred = []
    print('Starting Test...')
    y_pred = clf.predict(X_pred)

    f = open("final_results.txt","w")
    for i in y_pred:
        f.write(f'{i}\n')
    f.close()

    return y_pred





# Knn - Crossval
def KNN_cross(train_dat, test_dat, device, dtype):
    # params = {'reduction':[None, ['PCA', 0.95], ['PCA', 0.99], ['FA', 20], ['FA', 40]],
    #           'resample_val':[None,['max', True], ['min', True]],
    #           'k':[1, 2, 4, 8, 16, 32],
    #           'metric':['euclidean', 'cosine'],
    #           }
    
    params = {'reduction':[None, ['PCA', 0.95], ['PCA', 0.99], ['FA', 20],['FA', 30], ['FA', 40], ['FA', 50]],
              'resample_val':[None,[5000, True], ['max', True], ['min', False], ['min', True]],
              'k':[1, 2, 4, 16, 32],
              'metric':['euclidean', 'cosine', 'manhattan', 'chebyshev', 'minkowski'],
              }

    results = cross_val(train_dat=train_dat, clf_type='KNN', parameters=params, 
              device=device, dtype=dtype,)
    
    sorted_results = sorted(results.items(), reverse=True)
    f = open("sorted_results_KNN3.txt","w")
    f.write(str(sorted_results))
    f.close()


# #DT - Crossval
def DT_cross(train_dat, test_dat, device, dtype):
    params ={'reduction':[None, ['PCA', 0.95], ['PCA', 0.99], ['FA', 20], ['FA', 40]],
              'resample_val':[None,['max', True], ['min', True]],
              'func': ['gini', 'entropy'],  
              'features': [None, 'log2', 'auto'],
              'depths': [10, 100, 1000, 10000, None],
              'leafs': [10, 100, 1000, 10000, None],
              }
    results = cross_val(train_dat=train_dat, clf_type='DT', parameters=params, 
              device=device, dtype=dtype,)
    
    sorted_results = sorted(results.items(), reverse=True)
    f = open("sorted_results_DT2.txt","w")
    f.write(str(sorted_results))
    f.close()


#NB - Crossval
def NB_cross(train_dat, test_dat, device, dtype):
    params ={'reduction':[None, ['PCA', 0.95], ['PCA', 0.99], ['FA', 20], ['FA', 40]],
              'resample_val':[None,['max', True], ['min', True]],
              }
    results = cross_val(train_dat=train_dat, clf_type='NB', parameters=params, 
              device=device, dtype=dtype,)
    
    sorted_results = sorted(results.items(), reverse=True)
    f = open("sorted_results_NB2.txt","w")
    f.write(str(sorted_results))
    f.close()


#RF - Crossval
def RF_cross(train_dat, test_dat, device, dtype):
    params = {'reduction':[None, ['PCA', 0.95], ['PCA', 0.99], ['FA', 20], ['FA', 40]],
              'resample_val':[None,['max', True], ['min', True]],
              'n_estimators': [100, 10], 
              'criterion': ['gini', 'entropy'],  
              'min_samples_split': [2, 0.5, 10], 
              'min_samples_leaf': [1, 0.5, 10],
              'max_features': ['auto', 'log2', None],
              'max_depth': [10, 100, 1000, None],
              'max_leaf_nodes': [10, 100, 1000, None] ,
              'min_weight_fraction_leaf': [0.0], 
              'min_impurity_decrease': [0.0], 
              'ccp_alpha':[0.0], 
              'max_samples':[None],
              }
    params = {'reduction':[['FA', 50]],
              'resample_val':[['max', True]],
              'n_estimators': [100], 
              'criterion': ['gini', 'entropy'],  
              'min_samples_split': [2], 
              'min_samples_leaf': [1],
              'max_features': ['auto', 'log2', None],
              'max_depth': [1000, None],
              'max_leaf_nodes': [1000, None] ,
              'min_weight_fraction_leaf': [0.0], 
              'min_impurity_decrease': [0.0], 
              'ccp_alpha':[0.0], 
              'max_samples':[0.1, 0.5, 0.75, None],
              }
    
    results = cross_val(train_dat=train_dat, clf_type='RF', parameters=params, 
              device=device, dtype=dtype,)
    
    sorted_results = sorted(results.items(), reverse=True)
    f = open("sorted_results_RF4.txt","w")
    f.write(str(sorted_results))
    f.close()




device, dtype = use_gpu()
#Cross validation
train_dat = '../data/train.csv'
test_dat = '../data/test.csv'


#Cross val functions I ran
# KNN_cross(train_dat, test_dat, device, dtype)
# DT_cross(train_dat, test_dat, device, dtype)
# NB_cross(train_dat, test_dat, device, dtype)
# RF_cross(train_dat, test_dat, device, dtype)



# #Final Predictions
# params = {'reduction':['FA', 50],
#           'resample_val':['max', True],
#           'n_estimators': 100, 
#           'criterion': 'gini',
#           'min_samples_split': 2, 
#           'min_samples_leaf': 1,
#           'max_features': 'log2',
#           'max_depth': None,
#           'max_leaf_nodes': None ,
#           'min_weight_fraction_leaf': 0.0, 
#           'min_impurity_decrease': 0.0, 
#           'ccp_alpha':0.0, 
#           'max_samples':None,
#           }
# results = final_predict(train_dat=train_dat, test_dat=test_dat, clf_type='RF', param=params, 
#           device=device, dtype=dtype,)







#Unused code atm
#Write out best
# best = []
# f = open("best.txt","w")
# for i in sorted_results[0][1]:
#     best.append(i[0])
#     f.write(str(i[0])+'\n')
# f.close()

# #ANN - Crossval
# params ={'reduction':[None,],
#           'resample_val':[None,],
#           'n_input': [], 
#           'n_hidden': [], 
#           'n_layer': [], 
#           'n_out': [], 
#           'optimizer_lr': [('SGD', 0.01), ('Adam', 0.001)],
#           'dropout': [None, 0.10, 0.25],
#           'epochs': [60],
#           }
# results = cross_val(train_dat=train_dat, clf_type='ANN', parameters=params, 
#           device=device, dtype=dtype, cuda_nn=True)