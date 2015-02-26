# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 19:53:56 2015

@author: zhihuixie
"""

import pandas as pd

def feature_label_split(file1):
    """ this function read csv file and separate features and labels from file"""
    df = pd.read_csv(file1) # read data to pandas dataframe
    try:
        labels = df["label"].values # get labels
    except KeyError:
        labels = df["Label"].values
    features = df.iloc[:, 1:].values # get features
    return (features, labels)
def merge_files(f1, f2, name, test = False):
    """this function merge test.csv and standard output prediction results"""
    if test:
        df1 = pd.read_csv(f1)
        df2 = pd.read_csv(f2)
        df = df1.join(df2)
        df = df.iloc[:, 1:]
        df.to_csv("../data/test_labels_" + name + ".csv", index = False)    
    
if __name__ == "__main__":
    #feature_label_split("../data/train.csv")
    f1 = "../benchmark/knn_benchmark.csv"
    f2 = "../data/test.csv"
    name = "knn"
    merge_files(f1, f2, name, test = False)