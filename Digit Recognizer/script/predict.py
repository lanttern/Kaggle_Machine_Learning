# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 21:25:28 2015

@author: zhihuixie
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

PCA_COMPONENTS = 200

def predict(clf, features_train, labels_train, features_test, labels_test):
    """ this function is predicting digits in test file"""
    # apply pca
    pca = decomposition.PCA(n_components = PCA_COMPONENTS).fit(features_train)
    features_train = pca.transform(features_train)
    # fit train data
    clf.fit(features_train, labels_train)
    
    features_test = pca.transform(features_test)
    # prediction
    predictions = clf.predict(features_test)
    # calculate accuracy score
    score = accuracy_score(predictions, labels_test)
    
    return predictions, score
    
def output(predictions, loc):
    """ this function write prediction result to csv file"""
    df = pd.DataFrame(np.arange(1, len(predictions) + 1), columns = ["ImageId"])
    df["Label"] = predictions
    df.to_csv(loc, index = False)
    
if __name__ == "__main__":
    from load_data import feature_label_split
    #from sklearn.cross_validation import train_test_split
    from sklearn import tree
    is_knn = False
    features_train, labels_train = feature_label_split("../data/train.csv")
    if is_knn:
        features_test, labels_test = feature_label_split("../data/test_labels_knn.csv")
        clf = KNeighborsClassifier(n_neighbors = 10, algorithm = "kd_tree")
        loc = "../result/result_knn.csv"
    else:
        features_test, labels_test = feature_label_split("../data/test_labels_rf.csv")
        clf = tree.DecisionTreeClassifier(random_state = 100)
        loc = "../result/result_dt.csv"
        """
        clf = RandomForestClassifier(n_estimators = 1000)
        loc = "../result/result_rf.csv"
        """
        
    predictions, score = predict(clf, features_train, labels_train, features_test, labels_test)
    print is_knn
    output(predictions, loc)
    print score
    