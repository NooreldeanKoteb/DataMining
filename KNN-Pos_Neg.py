# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 16:03:14 2021

@author: Nooreldean Koteb
"""

#Hide useless SKlearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from math import sqrt
import numpy as np
import nltk
import scipy
import re
from numpy import dot
from numpy.linalg import norm
from itertools import product
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.sparse import csr_matrix
from collections import Counter

def preprocess(train_dat, test_dat=None, data_type=None, reduce=None,
               lemmatize=False, stemming=False, stop_words=False, 
               max_df=0.6, min_df=0.1, max_features=None, n_gram=(1,1)):
    
    if lemmatize and stemming:
        return print('Please pick Lemmatization or Stemming, not both!')
    
    X_test = []
    X_train = []
    y_train = []
    print('Reading Data...')
    
    if stop_words:
        print('Removing Stop words...')
        stop_w = stopwords.words('english')
    if stemming:
        print('Stemming data...')
        p_stemmer = PorterStemmer()
    if lemmatize:
        print('Lemmatizing data...')
        lemmatizer = WordNetLemmatizer() 

    with open(train_dat, "r", encoding="utf8") as f:
        for line in f:
            y_train.append(line[:2])
            
            line = cleanup(line)
            line = word_tokenize(line[3:-6].lower())
            
            if stop_words and stemming:
                line = stem_stop(p_stemmer, stop_w,line)  
                
            elif stop_words and lemmatize:
                line = lemma_stop(lemmatizer, stop_w,line)
            
            else:
                if stop_words:
                    line = stop(stop_w,line)  
                    
                if stemming:
                    line = stem(p_stemmer, line) 
                
                if lemmatize:
                    line = lemma(lemmatizer, line) 
            
            X_train.append(' '.join(line))
    
    
    if test_dat != None:
        with open(test_dat, "r", encoding="utf8") as f:
            for line in f:
                
                line = cleanup(line)
                line = word_tokenize(line[3:-6].lower())
                
                if stop_words and stemming:
                    line = stem_stop(p_stemmer, stop_w,line)  
                    
                elif stop_words and lemmatize:
                    line = lemma_stop(lemmatizer, stop_w,line)
                
                else:
                    if stop_words:
                        line = stop(stop_w,line)  
                        
                    if stemming:
                        line = stem(p_stemmer, line) 
                    
                    if lemmatize:
                        line = lemma(lemmatizer, line) 
                
                X_test.append(line)
    
    
    #Status Message
    if stemming and stop_words:
        print('Data has been read, Stop words were removed, and stemming is Complete!')
    elif lemmatize and stop_words:
        print('Data has been read, lemmatized, and Stop words were removed!')
    elif lemmatize:
        print('Data has been read, and lemmatized!')
    elif stop_words:
        print('Data has been read and Stop words were removed!')
    elif stemming:
        print('Data has been read and stemming is Complete!')
    else:
        print('Data has been read!')
    
    
    if reduce != None:
        X_test = X_test[:reduce]
        X_train = X_train[:reduce]
        y_train = y_train[:reduce]
    

    if data_type != None:
        print('Data is being converted to '+data_type+'...')
        if data_type == 'binary':
            X_vect = data_to_binary(X_train, max_df=max_df, min_df=min_df, max_features=max_features, n_gram=n_gram)
        elif data_type == 'raw_freq':
            X_vect = data_to_raw_freq(X_train, max_df=max_df, min_df=min_df, max_features=max_features, n_gram=n_gram)
        elif data_type == 'tf_idf':
            X_vect = data_to_tf_idf(X_train, max_df=max_df, min_df=min_df, max_features=max_features, n_gram=n_gram)
      
        X_train = X_vect.transform(X_train).toarray()
        if test_dat != None:
            X_test = X_vect.transform(X_test).toarray()
            
            
        print('Data has been converted to '+data_type+' form!')
    
    
    
    if data_type == 'tf_idf':
        print('Data is being Scaled...')
        
        sc_X = StandardScaler()
        sc_X.fit(X_train)
        X_train = sc_X.transform(X_train)
        
        if test_dat != None:
            X_test = sc_X.transform(X_test)
            
            
        print('Data has been scaled!')

    if data_type != 'None':
        y_train = np.array(y_train, np.float).reshape(len(y_train), 1)
        # y_train = y_train.astype(np.float)
        X_train = np.append(X_train, y_train, 1)

    return (X_train, X_test)

def cleanup(line):
    line = line.replace('<br />', ' ')
    line = re.sub(r'[^\w\s]',' ',line)
    line = re.sub(r'\d+',' ',line)
    return line

def stem(p_stemmer, line):
    new_line = []
    for word in line:
        new_line.append(word)
    return new_line
def stop(stop_w,line):
    new_line = []
    for word in line:
        if word not in stop_w:
            new_line.append(word)
    return new_line
def stem_stop(p_stemmer, stop_w, line):
    new_line = []
    for word in line:
        if word not in stop_w:
            new_line.append(p_stemmer.stem(word))
    return new_line

def lemma(lemmatizer, line):
    new_line = []
    for word in line:
        new_line.append(lemmatizer.lemmatize(word))
    return new_line

def lemma_stop(lemmatizer, stop_w, line):
    new_line = []
    for word in line:
        if word not in stop_w:
            new_line.append(lemmatizer.lemmatize(word))
    return new_line

def identityTokenizer(text):
    return text
    
def data_to_binary(data, max_df, min_df, max_features, n_gram):
    vectorizer = CountVectorizer(tokenizer=identityTokenizer, max_df=max_df, min_df=min_df, 
                                 max_features=max_features, ngram_range=n_gram, lowercase=False, binary=True)
    vectorizer.fit(data)
    return vectorizer

def data_to_raw_freq(data, max_df, min_df, max_features, n_gram):
    vectorizer = CountVectorizer(tokenizer=identityTokenizer, max_df=max_df, min_df=min_df, 
                                 max_features=max_features, ngram_range=n_gram, lowercase=False)
    vectorizer.fit(data)
    return vectorizer

def data_to_tf_idf(data, max_df, min_df, max_features, n_gram):
    vectorizer = TfidfVectorizer(tokenizer=identityTokenizer, max_df=max_df, min_df=min_df, 
                                 max_features=max_features, ngram_range=n_gram, lowercase=False)
    vectorizer.fit(data)
    return vectorizer

#I used tfidfvectorizer from sklearn because it was much better and faster.
#This is a from scratch implementation though.
def tfidf_vec(X):
    all_doc =[]
    for i in X:
        all_doc.extend(word_tokenize(i))
    
    doc_freq = dict(Counter(all_doc))
    N = len(doc_freq)
    
    vocab = []
    for i in doc_freq.keys():
        vocab.append(i)
        doc_freq[i] = len(doc_freq[i])
        
    
    tfidf = {}
    for i in range(num_doc):
        doc = dict(Counter(word_tokenize(X[i])))
        doc_N =len(doc)
        for j in np.unique(word_tokenize(X[i])):
            tf = doc[j]/doc_N
            df = doc_freq[j]
            idf = np.log(N/(df+1))
            tfidf[i, j] = tf*idf
    
    vector = np.zeros((len(X),len(vocab)))
    for i in tfidf:
        index = vocab.index(i[1])
        vector[i[0]][index] = tfidf[i]
        
    return vector
        
        
def evaluation(test, pred):
    cm = confusion_matrix(test, pred)
    print(cm)
    f1 = f1_score(test, pred)
    print(f1)
    acc = accuracy_score(test, pred)
    print(acc)
    return {'confusion_matrix': cm, 'f1_score': f1, 'accuracy_score': acc}


def euclidean_dist(vec1, vec2):
    return sqrt(np.sum(np.power(np.subtract(vec1, vec2), 2)))

def cosine_sim(vec1, vec2):
  return (dot(vec1, vec2) / (norm(vec1) * norm(vec2)))

def knn(train, test_vec, k, dist_fn='euclidean'):
    dist_list = []
    
    if dist_fn == 'euclidean':
        dist = euclidean_distances([test_vec], train[:,:-1])
        dist_list = np.append(train[:,-1].reshape(len(train[:,-1]), 1), dist.reshape(len(dist[0]), 1), 1)
        
    if dist_fn == 'cosine_sim':
        dist = cosine_similarity([test_vec], train[:,:-1])
        dist_list = np.append(train[:,-1].reshape(len(train[:,-1]), 1), dist.reshape(len(dist[0]), 1), 1)
    
    dist_list = sorted(dist_list, key=lambda x: x[1])
    
    neigh_list = []
    for neigh in range(k):
        neigh_list.append(dist_list[neigh][0])
    

    return neigh_list

def predict(train, test_vec, k, dist_fn='euclidean'):
    neighbors = knn(train, test_vec, k, dist_fn=dist_fn)
    
    return max(set(neighbors), key=neighbors.count)
        
def cross_val(train_dat, parameters, reduce=None):
    keys, values = zip(*parameters.items())
    combinations = []
    for val in product(*values):
        combinations.append(dict(zip(keys, val)))
    
    params = []
    for param in combinations:
        if param['stemming'] == True and param['lemmatize'] == True:
            continue
        elif param['stemming'] == False and param['lemmatize'] == False:
            continue
        else:
            params.append(param)
    
    
    results = {}
    X_train = []    
    cur_data = []
    for param in params:
        print(param)
        data_type = param['data_type']
        stemming = param['stemming']
        lemmatize = param['lemmatize']
        stop_words = param['stop_words']
        max_df = param['max_df']
        min_df = param['min_df']
        max_features = param['max_features']
        n_gram = param['n-gram']
        
        k = param['k']
        dist_fn = param['dist_fn']
        
        #speed some preprocessing time
        if cur_data == [] or cur_data != [data_type, stemming, lemmatize, stop_words, max_df, min_df, max_features]:
            cur_data = [data_type, stemming, lemmatize, stop_words, max_df, min_df, max_features]

            X_train, _ = preprocess(train_dat=train_dat, test_dat=None, 
                                         data_type=data_type, stemming=stemming, n_gram=n_gram,
                                         lemmatize=lemmatize, stop_words=stop_words, reduce=reduce,
                                         max_df=max_df, min_df=min_df, max_features=max_features)
            
        #split data
        cross_train, cross_test = train_test_split(X_train, test_size=0.3, shuffle=True)
     
        
        X_test = cross_test[:, :-1]
        y_test = cross_test[:, -1].T
        y_test.reshape(1, len(y_test))
        y_pred = []
        progress = 0
        completion = len(y_test)
        
        print('Starting Test...')
        for test_row in X_test:            
            y_pred.append(predict(cross_train, test_row, k, dist_fn=dist_fn))
            
            progress+=1
            current = (progress/completion)
            if current == 0.2:
                print('20% Complete')
            elif current == 0.4:
                print('40% Complete')
            elif current == 0.6:
                print('60% Complete')
            elif current == 0.8:
                print('80% Complete')
            elif current == 1.0:
                print('100% Complete')        
        
        scores = evaluation(y_test, y_pred)
        
        if scores['accuracy_score'] in results:
            results[scores['accuracy_score']].append([param, scores])
        else:
            results[scores['accuracy_score']] = [[param, scores]]
        
    return results

def final_predict(train_dat, test_dat, param, reduce=None):
    data_type = param['data_type']
    stemming = param['stemming']
    lemmatize = param['lemmatize']
    stop_words = param['stop_words']
    max_df = param['max_df']
    min_df = param['min_df']
    max_features = param['max_features']
    # n_gram = param['n-gram']
    
    k = param['k']
    dist_fn = param['dist_fn']
    
    #speed some preprocessing time
    if stemming and lemmatize:
        return print('Choose Stemming or lemmatizing!')

    X_train, X_test = preprocess(train_dat=train_dat, test_dat=test_dat, 
                                 data_type=data_type, stemming=stemming,
                                 lemmatize=lemmatize, stop_words=stop_words, reduce=reduce,
                                 max_df=max_df, min_df=min_df, max_features=max_features)
        

    y_pred = []
    progress = 0
    completion = len(X_test)
    print('Starting Test...')
    for test_row in X_test:            
        y_pred.append(predict(X_train[:5000], test_row, k, dist_fn=dist_fn))
        print(test_row)
        progress+=1
        current = (progress/completion)
        if current == 0.2:
            print('20% Complete')
        elif current == 0.4:
            print('40% Complete')
        elif current == 0.6:
            print('60% Complete')
        elif current == 0.8:
            print('80% Complete')
        elif current == 1.0:
            print('100% Complete')        
        

    f = open("final_results.txt","w")
    for i in y_pred:
        if i == 1:
            f.write('+1\n')  
        else:
            f.write('-1\n')  
    f.close()
        
    return y_pred



#####################################################################
#Cross validation
# parameters = {
#     'data_type': ['tf_idf', 'raw_freq', 'binary'],
#     'stop_words': [True, False],
#     'stemming': [False, True],
#     'lemmatize': [True, False],
#     'n-gram': [(1,1), (2,2), (3,3)],
#     'max_df':[0.6, 0.7, 0.8],
#     'min_df': [0, 0.1, 0.2],
#     'max_features': [None, 1500, 50000],
#     'k': [2, 8, 64, 128],
#     'dist_fn': ['euclidean', 'cosine_sim'],
#     }

# results = cross_val("train.dat", parameters)

# sorted_results = sorted(results.items(), reverse=True)

# f = open("results.txt","w")
# f.write(str(results))
# f.close()
# f = open("sorted_results.txt","w")
# f.write(str(sorted_results))
# f.close()

 
#####################################################################
# Final predictions        
final_parameters = {
    'data_type': 'tf_idf',
    'stop_words': True,
    'stemming': False,
    'lemmatize': True,
    'n-gram': (3,3),
    'max_df':0.6,
    'min_df': 0,
    'max_features': None,
    'k': 64,
    'dist_fn': 'euclidean',
    }


y_pred = final_predict(train_dat='train.dat', test_dat='test.dat', param=final_parameters)











