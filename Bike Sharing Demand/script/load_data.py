# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 2015

@author: zhihuixie
"""

import pandas as pd

def add_feature(file1, loc):
    df = pd.read_csv(file1) # read data to pandas dataframe
    df ["hour"] = [int(date[11:13]) for date in df["datetime"].values]
    df.to_csv(loc, index = False)
    
def feature_label_split(file1):
    """ this function read csv file and separate features and labels from file"""
    unused_features = ["casual", "registered"]
    df = pd.read_csv(file1) # read data to pandas dataframe
    try:
        labels = df["count"].values # get labels
        df.drop("count", axis = 1, inplace = True)
    except KeyError:
        labels = []
    for feature in unused_features:
        try:
            df.drop(feature, axis = 1, inplace = True)
        except ValueError:
            continue
    features = df.iloc[:, 1:].values # get features
    return (features, labels)
 
    
if __name__ == "__main__":
    #feature_label_split("../data/train.csv")
    files = ["../data/train.csv", "../data/test.csv"]
    locs = ["../data/train_add_feature.csv", "../data/test_add_feature.csv"]
    for (file1, loc) in zip(files, locs):
        add_feature(file1, loc)
    features, labels = feature_label_split(locs[0])
    print len(features[1])#, labels
    