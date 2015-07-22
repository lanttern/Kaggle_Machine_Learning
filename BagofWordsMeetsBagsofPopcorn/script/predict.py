# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:27:09 2015

@author: zhihuixie
"""
# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup             
import re
from nltk.corpus import stopwords

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   
    
if __name__ == "__main__":
    import pandas as pd     
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.ensemble import RandomForestClassifier  
    from sklearn.tree import DecisionTreeClassifier 
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.cross_validation import train_test_split
    
    train = pd.read_csv("../data/labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
                    
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["review"].size

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    print "Cleaning and parsing the training set movie reviews...\n"
    
    for i in xrange( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
        if( (i+1)%1000 == 0 ):
            print "Review %d of %d\n" % ( i+1, num_reviews )                                                                    
        clean_train_reviews.append( review_to_words( train["review"][i] ))
        
    
    print "Creating the bag of words...\n"


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    train_data_features = train_data_features.toarray()
    
    print "split training and test dataset for cross validation..."
    forest = RandomForestClassifier(n_estimators = 100)  
    DC = DecisionTreeClassifier()
    svc = SVC(C = 10.0)

    features_train, features_test, labels_train, labels_test = train_test_split\
         (train_data_features, train["sentiment"], test_size = 0.33, random_state = 42)
    print "Training the random forest..."
    clf1 = forest.fit(features_train, labels_train)
    print "Predict..."
    pred1 = forest.predict(features_test)
    print "Training the svc..."
    clf2 = svc.fit(features_train, labels_train)
    print "Predict..."
    pred2 = svc.predict(features_test)
    
    print 0.841939393939, accuracy_score(pred2, labels_test)
    
    
    """
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100) 

    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train["sentiment"] )
    
    
    # Read the test data
    test = pd.read_csv("../data/testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )

    # Verify that there are 25,000 rows and 2 columns
    print test.shape

    # Create an empty list and append the clean reviews one by one
    num_reviews = len(test["review"])
    clean_test_reviews = [] 

    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(0,num_reviews):
        if( (i+1) % 1000 == 0 ):
            print "Review %d of %d\n" % (i+1, num_reviews)
        clean_review = review_to_words( test["review"][i] )
        clean_test_reviews.append( clean_review )

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

    # Use pandas to write the comma-separated output file
    output.to_csv( "../data/Bag_of_Words_model.csv", index=False, quoting=3 )
    """