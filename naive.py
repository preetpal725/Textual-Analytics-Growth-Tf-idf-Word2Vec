#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 22:46:46 2019

@author: jiangzhaobo
"""

import pandas as pd
import numpy as np
#import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
import jieba

import string, nltk, csv
from nltk.corpus import stopwords

data=pd.read_csv('Growth.csv') 
''' Converting data to two different lists'''
data=np.array(data.loc[:,:])
list_sentences = []
list_classifier = []
for i in range (0,data.shape[0]):
    temp_clasifier = data[i][0]
    temp_sentence = data[i][1]
    list_classifier.append(temp_clasifier)        
    list_sentences.append(str(temp_sentence).lower())
#print(len(list_sentences))  
#print(len(list_classifier))

''' Removing the punctuation '''
list_sentences_no_punc = []
for sent in list_sentences:
    for c in string.punctuation:
        sent = sent.replace(c," ")
    list_sentences_no_punc.append(sent)
#print(list_sentences_no_punc)

''' Lemmatization '''
lemma = nltk.wordnet.WordNetLemmatizer()
lematized_list = []
for sent in list_sentences_no_punc:
    temp_list = []
    temp_line_list = word_tokenize(sent)
    for word in temp_line_list:
        temp_list.append(lemma.lemmatize(word, pos = "v").strip())
    lematized_line = ' '.join(temp_list)  
    lematized_list.append(lematized_line)
#print(lematized_list)

''' Removing stopwords '''
list_without_stopwords = []
list_stopwords = stopwords.words('english')
for sent in lematized_list:
    temp_word_list = word_tokenize(sent)
    temp_list = []
    for word in temp_word_list:
        if word not in list_stopwords:
            temp_list.append(word)
    line_without_stopwords = ' '.join(temp_list)
    list_without_stopwords.append(line_without_stopwords)       
#print(list_without_stopwords)
#print(len(list_without_stopwords))


''' Writing a csv file '''
with open('Lemmatized_file.csv', mode='w', encoding = "utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Classification", "Sentences"])
    writer.writerows(zip(list_classifier, lematized_list))
#csv_file.close()

''' Reading the file '''
data = pd.read_csv('Lemmatized_file.csv' )

X=data['Sentences']
X=[i.split() for i in X]
X[:2]
import gensim
model = gensim.models.Word2Vec(X,min_count =5,window =8,size=100)   
embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))

print('Found %s word vectors.' % len(embeddings_index))
model['growth']
def sent2vec(s):
    words = str(s).lower()
    words = jieba.lcut(words)
#    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.Classification.values)
xtrain, xvalid, ytrain, yvalid = train_test_split(data.Sentences.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)
xtrain_w2v = [sent2vec(x) for x in tqdm(xtrain)]
xvalid_w2v = [sent2vec(x) for x in tqdm(xvalid)]

xtrain_w2v = np.array(xtrain_w2v)
xvalid_w2v = np.array(xvalid_w2v)




#Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB() 
clf.fit(xtrain_w2v,ytrain) 
predictions = clf.predict(xvalid_w2v)
print(predictions)
type(yvalid)
actual= np.array(yvalid.tolist())
actual
rere=0
reir=0
irre=0
irir=0
nnn=len(actual)
for ii in range(0,nnn):
    if actual[ii]==1:
        if predictions[ii]==1:
            rere+=1
        else:
            irre+=1
    else:
        if predictions[ii]==1:
            reir+=1
        else:
            irir+=1
#The accurary of Naive Bays
accur=(irir+rere)/(irir+rere+reir+irre)
print('The accuracy of Naive Bayes classifier is :',+accur)

#The relevant recall
reca=(rere)/(rere+reir)
print('The relevant recall of Naive Bayes classifier is :',+reca)
#The irrelevant recall
irca=(irir)/(irre+irir)
print('The irrelevant recall of Naive Bayes classifier is :',+irca)
#The relevant precision
repr=(rere)/(rere+irre)
print('The relevant precision of Naive Bayes classifier is :',+repr)
#The irrelevant precision
irpr=(irir)/(irir+reir)
print('The irrelevant precision of Naive Bayes classifier is :',+irpr)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(actual, predictions)
print(fpr)
print(tpr)
print(thresholds)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(actual, predictions)
print('The auc:',auc)
def class_logloss(actual, predicted, eps=1e-15):
    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
print ("logloss: %0.3f " % class_logloss(yvalid, predictions))
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.title("Naive Bayes-growth")
#plt.title("Naive Bayes-opportunity-auc=%.4f"%(auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.fill_between(fpr, tpr, where=(tpr>=0), color='blue', alpha=0.5)
plt.show()
