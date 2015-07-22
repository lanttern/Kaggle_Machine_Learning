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
if __name__ == '__main__':

    # Load the training file
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    df = pd.DataFrame()
    df["id"] = test.id
    # we dont need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    # create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)
    
    product_title = list(train["product_title"].values)
    product_description = list(train["product_description"].values)

    for i in range(len(product_description)):  #replace missing description with product title
        if type(i) != str:
            product_description[i] = product_title[i]
    train["product_description"] = product_description
    # do some lambda magic on text columns
    traindata = list(train.apply(lambda x:'%s %s' % (x['query'], x['product_title']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s' % (x['query'], x['product_title']),axis=1))
    
    # the infamous tfidf vectorizer (Do you remember this one?)
    tfv = TfidfVectorizer(min_df=10,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    
    # Fit TFIDF
    tfv.fit(traindata)
    X = tfv.transform(traindata)
    print X.shape[1]
    """
    .toarray()
    print len(X), len(X[0])
    for i in X[0]:
        if i != 0.0:
           print i
    """
    #features_train, features_test, labels_train, labels_test = train_test_split\
     #    (X, y, test_size = 0.33, random_state = 42)
         
    X_test = tfv.transform(testdata)
    print X_test.shape[1]

    clf = linear_model.LogisticRegression(C=1,tol = 0.0001)  
    """
    logistic = linear_model.LogisticRegression()    
    rbm = neural_network.BernoulliRBM(random_state = 0, verbose = True)
    clf = Pipeline(steps = [("rbm", rbm), ("logistic", logistic)])
    rbm.learning_rate = 0.05
    rbm.n_iter = 20
    rbm.n_components = 100
    logistic.C = 1
    logistic.tol = 0.0001
    """
    """
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print accuracy_score(pred, labels_test)
    """
    """
    clf.fit(X,y)
    pred = clf.predict(X_test)
    df["prediction"] = pred
    df.to_csv("../data/test_result.csv", index = False)
    """
    