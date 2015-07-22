# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:16:48 2015

@author: zhihuixie
"""
from similarity import relevance, count_word
from text_transform import transform
from similarity_test import score

def extract_features_labels(file1):
    """transform and extract features, labels"""
    data = pd.read_csv(file1)
    feature1 = data["feature1"]
    feature2 = data["feature2"]
    feature3 = data["feature3"]
    feature4 = data["feature4"]
    feature5 = data["feature5"]
    feature6 = data["feature6"]
    features = zip(feature1, feature2, feature3, feature4, feature5, feature6)
    features = np.array(features)
    return features

def output(pred, data_test, path):
    """write output to csv file"""
    df = pd.DataFrame()
    df["id"] = data_test["id"]
    #pred = [int(round(num)) for num in pred]
    df["prediction"] = pred
    df.to_csv(path, index = False)

if __name__ == "__main__":
    import pandas as pd
    from sklearn.cross_validation import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import linear_model, tree, ensemble, cluster, svm, naive_bayes
    from sklearn.svm import SVC
    from sklearn import grid_search, metrics
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction import text
    from nltk.stem.porter import *
    import re
    from bs4 import BeautifulSoup
    import numpy as np
    #from sklearn import preprocessing 
    train = pd.read_csv("../data/train.csv", sep = ",").fillna("")
    test = pd.read_csv("../data/test.csv", sep = ",").fillna("")
    
    sw=[]
    #stopwords tweak - more overhead
    stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
    stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
    for stw in stop_words:
        sw.append("q"+stw)
        sw.append("z"+stw)
    stop_words = text.ENGLISH_STOP_WORDS.union(sw)


    #remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
    stemmer = PorterStemmer()
    ## Stemming functionality
    class stemmerUtility(object):
        """Stemming functionality"""
        @staticmethod
        def stemPorter(review_text):
            porter = PorterStemmer()
            preprocessed_docs = []
            for doc in review_text:
                final_doc = []
                for word in doc:
                    final_doc.append(porter.stem(word))
                    #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
                preprocessed_docs.append(final_doc)
            return preprocessed_docs
    s_data = []
    t_data = []
    s_labels = []
    
    print "cleaning dataset..."
    for i in range(len(train.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
        s_labels.append(str(train["median_relevance"][i]))
    for i in range(len(test.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        t_data.append(s)
    
    print "fit and transform training dataset..."    
    tfv = TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')
    tfv.fit(s_data)    
    features1 = extract_features_labels("../data/features.csv")
    features2 = transform(s_data, tfv)
    features = np.concatenate((features1, features2), axis = 1)

    print "fit and transform test dataset..."
    features_test1 = extract_features_labels("../data/features_test.csv")
    features_test2 = transform(t_data, tfv)
    features_test = np.concatenate((features_test1, features_test2), axis = 1)
    labels = train["median_relevance"].values 
    
    print "Training model..."
    #create sklearn pipeline, fit all, and predit test data
    clf = Pipeline([ 
    ('svd', TruncatedSVD(n_components=400, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
    ('svm', SVC(C=10.0, kernel='rbf', degree=5, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    clf.fit(features, labels)    
    
    print "Predict outcomes..."
    t_labels = clf.predict(features_test)
    
    print "Write to file..."    
    path = "../data/result1.csv"
    output(t_labels, test, path)
    print "Completed!!!"
    