# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:40:27 2015

@author: zhihuixie
"""
import pandas as pd
import numpy as np
from text_transform import transform
from scorer import quadratic_weighted_kappa


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
    
def pred(clf, features_train, labels_train, features_test):
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    return pred

def validation (pred, labels_test):
    """Valide classifier"""
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    (score1, score2, score3) = (accuracy_score (pred, \
    labels_test), f1_score (pred, labels_test), recall_score (pred, labels_test))
    
    return zip(["accuracy_score", "f1_score", "recall_score"], \
           [score1, score2, score3])

def output(pred, data_test, path):
    """write output to csv file"""
    df = pd.DataFrame()
    df["id"] = data_test["id"]
    #pred = [int(round(num)) for num in pred]
    df["prediction"] = pred
    df.to_csv(path, index = False)
    
    
if __name__ == "__main__":
    from sklearn.cross_validation import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import linear_model, tree, ensemble, cluster, svm, naive_bayes
    from sklearn import grid_search, metrics
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction import text
    from nltk.stem.porter import *
    import re
    from bs4 import BeautifulSoup

    #from sklearn import preprocessing 
    print "load data..."
    data = pd.read_csv("../data/train.csv", sep = ",")
    data_test = pd.read_csv("../data/test.csv", sep = ",")
    data_text = list(data.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    data_test_text = list(data_test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
  
    tfv = TfidfVectorizer(min_df=3,max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    
    # Fit TFIDF
    print "fit and transform data..."
    tfv.fit(data_text)    
    
    print "extract features and labels..."
    features1 = extract_features_labels("../data/features.csv")
    features2 = transform(data_text, tfv)
    features = np.concatenate((features1, features2), axis = 1)

    
    features_test1 = extract_features_labels("../data/features_test.csv")
    features_test2 = transform(data_test_text, tfv)
    features_test = np.concatenate((features_test1, features_test2), axis = 1)
    
    labels = data["median_relevance"].values 

    """
    features_train, features_test, labels_train, labels_test = train_test_split\
         (features, labels, test_size = 0.33, random_state = 42)
    """
    
    # Initialize SVD
    svd = TruncatedSVD()
    # Initialize the standard scaler 
    scl = StandardScaler()
    # We will use SVM here..
    svm_model = svm.SVC()
    # Create the pipeline 
    clf = Pipeline([('svd', svd), ('scl', scl),('svm', svm_model)])
    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'svd__n_components': [400],
                  'svm__C': [10]}
    # Kappa Scorer 
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)
    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, 
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2, scoring=kappa_scorer)
    # Fit Grid Search Model
   # model.fit(features_train, labels_train)
    print "fit search model..."                             
    model.fit(features, labels)
    print "fit completed!!"
    
    #print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
    	print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    # Get best model
    best_model = model.best_estimator_
    #predition = pred(best_model, features_train, labels_train, features_test)
    
    #clf2 = linear_model.SGDClassifier()    
    #predition2 = pred(clf2, features_train, labels_train, features_test)
    #predition2 = pred(clf2, features, labels, features_test)
    print "train and predict outcomes..."
    prediction = pred(best_model, features, labels, features_test)
      
    #final_pred = []    
    #for i in range(len(predition)):
    #    final_pred.append(int((predition[i] + predition2[i])/2))
    #print final_pred, labels_test
    #print validation (final_pred, labels_test)
       
    print "write to file..."
    path = "../data/result.csv"
    output(prediction, data_test, path)
    print "completed!!"
    
    