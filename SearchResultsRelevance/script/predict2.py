# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:40:27 2015

@author: zhihuixie
"""
import pandas as pd
import numpy as np


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
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from sklearn import decomposition, pipeline, metrics, grid_search
    from nltk.stem.porter import *
    import re
    from bs4 import BeautifulSoup
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.feature_extraction import text
    import string
    from sklearn.pipeline import Pipeline, FeatureUnion
    from text_transform import transform
    
    #from sklearn import preprocessing 
    print "load data..."
    # Load the training file
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
    
    
    print "Extract features.."      
    tfv = TfidfVectorizer(min_df=3,max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    tfv.fit(s_data)    
    features1 = extract_features_labels("../data/features.csv")
    features2 = transform(s_data, tfv)
    #features = np.concatenate((features1, features2), axis = 1)

    print "fit and transform test dataset..."
    features_test1 = extract_features_labels("../data/features_test.csv")
    features_test2 = transform(t_data, tfv)
    #features_test = np.concatenate((features_test1, features_test2), axis = 1)
    labels = train["median_relevance"].values 
    
    # Initialize SVD
    svd = TruncatedSVD(n_components=400)
    from sklearn.metrics.pairwise import linear_kernel
    class FeatureInserter():
        
        def __init__(self):
            pass
        
        def transform(self, X, y=None):
            distances = []
            quasi_jaccard = []
            print(len(distances), X.shape)
            
            for row in X.tocsr():
                row=row.toarray().ravel()
                cos_distance = linear_kernel(row[:row.shape[0]/2], row[row.shape[0]/2:])
                distances.append(cos_distance[0])
                intersect = row[:row.shape[0]/2].dot(row[row.shape[0]/2:])
                union = (row[:row.shape[0]/2]+row[row.shape[0]/2:]).dot((row[:row.shape[0]/2]+row[row.shape[0]/2:]))
                quasi_jaccard.append(1.0*intersect/union)
                
            print(len(distances), X.shape)
            print(distances[:10])
            
            #X = scipy.sparse.hstack([X, distances])
            return np.matrix([x for x in zip(distances, quasi_jaccard)])
            
        def fit(self, X,y):
            return self
            
        
        def fit_transform(self, X, y, **fit_params):
            self.fit(X,y)
            return self.transform(X)
    
    # Initialize the standard scaler 
    scl = StandardScaler()
    
    # We will use SVM here..
    svm_model = SVC(C=10.)
    
    # Create the pipeline 
    model = pipeline.Pipeline([('UnionInput', FeatureUnion([('svd', svd), ('dense_features', FeatureInserter())])),
    						 ('scl', scl),
                    	     ('svm', svm_model)])
    print "train and predict outcomes..."
    prediction = pred(model, features2, labels, features_test2)
    
      
    #final_pred = []    
    #for i in range(len(predition)):
    #    final_pred.append(int((predition[i] + predition2[i])/2))
    #print final_pred, labels_test
    #print validation (final_pred, labels_test)
       
    print "write to file..."
    path = "../data/result2.csv"
    output(prediction, test, path)
    print "completed!!"
    
    