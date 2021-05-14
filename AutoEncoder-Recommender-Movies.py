#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:40:08 2021

@author: Nooreldean Koteb
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from collections import defaultdict 
from sklearn.model_selection import train_test_split

class SAE(nn.Module):
    def __init__(self, inp, hidden, layers=2):
        super(SAE, self).__init__()
        self.line1 = nn.Linear(inp, hidden)
        self.line2 = nn.Linear(hidden, hidden//2)
        self.line3 = nn.Linear(hidden//2, hidden)
        # self.line4 = nn.Linear(hidden//4, hidden//2)
        # self.line5 = nn.Linear(hidden//2, hidden)
        # self.line6 = nn.Linear(hidden//4, hidden//2)
        # self.line7 = nn.Linear(hidden//2, hidden)

        self.line4 = nn.Linear(hidden, inp)
        
        self.activation = nn.Sigmoid()
        
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x):
        x = self.activation(self.line1(x))
        # x = self.dropout(x)
        
        x = self.activation(self.line2(x))
        # x = self.dropout(x)
        
        x = self.activation(self.line3(x))
        # x = self.dropout(x)
        
        # x = self.activation(self.line4(x))
        # x = self.dropout(x)
        
        # x = self.activation(self.line5(x))
        # x = self.dropout(x)
        
        # x = self.activation(self.line6(x))
        # x = self.dropout(x)
        
        # x = self.activation(self.line7(x))
        # # x = self.dropout(x)
        
        x = self.line4 (x)
        
        return x

#Use GPU if available
def use_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    return device

#Used For default dictionary
def def_value():
    return 'None'



def preprocess(train_data, test_data):
    #UserID, MovieID
    test = np.array(pd.read_csv(test_data, sep = ' ',  header = 0), dtype=int)
    #UserID, MovieID, Rating
    train = np.array(pd.read_csv(train_data, dtype=float, sep = ' ', header = 0, encoding = 'latin-1'), dtype=float)
    
    users_dic = defaultdict(def_value)
    movies_dic = defaultdict(def_value)
    
    users_list = list(train[:, 0])+list(test[:, 0])
    movies_list = list(train[:, 1])+list(test[:, 1])
    
    key1 = 0
    for i in users_list:
        if i not in users_dic.keys():
            key1+=1
            users_dic[i] = key1
    

    key2 = 0
    for i in movies_list:
        if i not in movies_dic.keys():
            key2+=1
            movies_dic[i] = key2
    
    
    n_users = key1       
    n_movies = key2
    
    return (test, train, users_dic, movies_dic, users_list, movies_list, n_users, n_movies)
    
#Create User Profiles from data
def profile(dat, device, n_movies, movies_dic, user_dic):
    profiles = []
    for user in users_dic.keys():
        movies = dat[:, 1][dat[:, 0] == user]
        ratings = dat[:, 2][dat[:, 0] == user]
        
        movie_keys = []
        for i in movies:
            movie_keys.append(movies_dic[i])
        
        profile = np.zeros(n_movies)
        profile[movie_keys] = ratings
        
        
        profiles.append(profile)
    return torch.tensor(profiles, dtype=torch.float, device=device)
    

def train_model(device, n_users, n_movies, X_train, 
          epochs=50, layers=2, hidden=20, optimizer=('RMS', 0.01, 0.5)):
    
    model = SAE(n_movies, hidden=hidden, layers=layers)
    if device.type == 'cuda':
        model.cuda()
    

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr = optimizer[1], weight_decay = optimizer[2])#


    for epoch in range(1, epochs + 1):
        RMSE = 0
        step = 0
        
        #Loop through all users
        for user in range(n_users):
            expected = Variable(X_train[user]).unsqueeze(0).clone()
            
            if torch.sum(expected.data > 0) > 0:
                result = model(Variable(X_train[user]).unsqueeze(0))
                
                expected.require_grad = False

                result[expected == 0] = 0                
                loss = criterion(result, expected)
                rated = expected.data > 0
                mean_corrector = n_movies/float(torch.sum(rated) + .00000000001)
                
                loss.backward()
                
                #RMSE
                RMSE += torch.sqrt(loss.data * mean_corrector)
                optimizer.step()
            
        print('epoch: '+str(epoch)+'/'+str(epochs)+' train loss: '+str(RMSE/step))        
    return model

# def test_model(model, n_users, n_movies, X_train, X_test):
#     RMSE = 0
#     step = 0

#     for user in range(n_users):
#         expected = Variable(X_test[user]).unsqueeze(0).clone()
        
#         if torch.sum(expected.data > 0) > 0:
#             result = model(Variable(X_train[user]).unsqueeze(0))
            
#             expected.require_grad = False

#             result[expected == 0] = 0                
#             loss = criterion(result, expected)
#             rated = expected.data > 0
#             mean_corrector = n_movies/float(torch.sum(rated) + .00000000001)
            
#             loss.backward()
            
#             #RMSE
#             RMSE += torch.sqrt(loss.data * mean_corrector)
#             optimizer.step()
    
#     #Print normalized test loss
#     print('test loss: '+str(RMSE/step))        
#     return test_loss/s
    
def pred_model(model,X_train, n_users, pred_dat, users_dic, movies_dic):    
    #Pred results
    raw_results = []
    for user in range(n_users):
        result = model(Variable(X_train[user]).unsqueeze(0))
        raw_results.append(result.data.cpu().numpy())
    
    
    results = []
    t = 0
    for i in pred_dat:
        try:
            results.append(f'{raw_results[users_dic[i[0]]][0][movies_dic[i[1]]] :.1f}') 
        except:
            t+=1
            results.append('2.5')
    
    return results
    
def write(results, outfile):      
    with open(outfile,"w") as f:
        for i in results:
            print(i)
            f.write(f'{i}\n')  
            
        f.close()        


# In[2]:


device = use_gpu()


# In[3]:


pred_dat, train, users_dic, movies_dic, users_list, movies_list, n_users, n_movies = preprocess('additional_files/train.dat', 'test.dat')


# In[4]:


X_train = profile(train, device, n_movies, movies_dic, users_dic)

X_train


# In[5]:


model = train_model(device, n_users, n_movies, X_train, epochs=100, layers=2, 
                    hidden=20, optimizer=('RMS', 0.00001, 0.5))



results = pred_model(model, X_train, n_users, pred_dat, users_dic, movies_dic)

results


# In[6]:


write(results, '7.txt')


# In[ ]:




