# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 2015

@author: zhihuixie
"""
from sklearn import tree, linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd


def predict(clf, features_train, labels_train, features_test, labels_test):
    """ this function is predicting digits in test file"""
    
    # fit train data
    clf.fit(features_train, labels_train)
    
    # prediction
    predictions = clf.predict(features_test)
    #predictions = [int(pred) for pred in predictions]
    #print predictions
    # calculate accuracy score
    #score = accuracy_score(predictions, labels_test)
    
    return predictions#, score
    
def output(test_file, predictions, loc):
    """ this function write prediction result to csv file"""
    df = pd.DataFrame()
    df["datetime"] = pd.read_csv(test_file)["datetime"]
    df["count"] = predictions
    df.to_csv(loc, index = False)    

if __name__ == "__main__":
    from load_data import feature_label_split
    from sklearn import preprocessing
    from sklearn import linear_model
    from sklearn import cluster, svm
    from sklearn.neighbors import KNeighborsRegressor
    
    features_train, labels_train = feature_label_split("../data/train_add_feature.csv")
    features_test, labels_test = feature_label_split("../data/test_add_feature.csv")
    #features_train, features_test, labels_train, labels_test = train_test_split\
     #   (features_train, labels_train, test_size = 0.4, random_state = 42)
    
    clf_t = tree.DecisionTreeClassifier(random_state = 100, max_depth = 1000)

    clf = clf_t
    test_file = "../data/test_add_feature.csv"
    loc = "../result/result.csv"
    predictions= predict(clf, features_train, labels_train, features_test, labels_test)
    #print scores
    output(test_file,predictions, loc)

    