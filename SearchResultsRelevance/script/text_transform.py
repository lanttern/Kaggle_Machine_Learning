# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:54:56 2015

@author: zhihuixie
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, tree, ensemble, cluster, svm
from sklearn import neural_network
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def transform(data_text, tfv):
    # do some lambda magic on text columns

    # the infamous tfidf vectorizer (Do you remember this one?)
    features = tfv.transform(data_text).toarray()
    return features
    
    
"""    
if __name__ == '__main__':
    
    #from sklearn import preprocessing 
    data = pd.read_csv("../data/train.csv", sep = ",").fillna("")
    query = list(data["query"].values)
    product_title = list(data["product_title"].values)
    product_description = list(data["product_description"].values)
    
    #data_test = pd.read_csv("../data/test.csv", sep = ",").fillna("")
    
    
    for i in range(len(product_description)):  #replace missing description with product title
        if product_description[i] == "" and product_title[i] != "":
            product_description[i] = product_title[i]
        if product_title[i] == "" and product_description[i] != "":
            product_title[i] = product_description[i]
        if product_title[i] == product_description[i] == "":
            product_title[i] = product_description[i] = "a"
        if query[i] == "":
            query[i] = "a"
            
    data["query"] = query
    data["product_title"] = product_title
    data["product_description"] = product_description
    
    data_text = list(data.apply(lambda x:'%s %s %s' % (x['query'], x['product_title'], x["product_description"]),axis=1))
    
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    
    # Fit TFIDF
    tfv.fit(data_text)    
    
    #data_test_text = list(data_test.apply(lambda x:'%s %s' % (x['query'], x['product_title']),axis=1))
    
    features = transform(data_text, tfv)
    df = pd.DataFrame(features)
    df.to_csv("../data/test1.csv")
"""