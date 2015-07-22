# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 21:57:48 2015

@author: zhihuixie
"""
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import pandas as pd
import numpy as np

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w.decode("utf8")  for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)
    
#nltk.download()    
# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    review = review.decode("utf8")    
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    
    return sentences

def score (query, product, model = Word2Vec.load("300features_40minwords_10context")):
    from sklearn.preprocessing import MinMaxScaler
    features = []
    for index in range(len(query)):
        review = product[index]
        query_word = query[index]
        sentences = review_to_sentences( review, tokenizer, remove_stopwords=False )
        if not sentences[0]:
            try:
               sentences[0] = sentences[1]
            except IndexError:
               sentences[0] = ["a"]
        try:
            if model.n_similarity(query_word.split(), sentences[0]) == np.nan:
                print review, query_word, model.n_similarity(query_word.split(), sentences[0])
            features.append (model.n_similarity(query_word.split(), sentences[0]))
        except KeyError:
            features.append(0.0) 
 
    scale = MinMaxScaler()
    features = scale.fit_transform(features)
    return features

   
if __name__ == "__main__":
    # run the following code to setup training features file
    """
    train = pd.read_csv("../data/train.csv", sep = ",").fillna("")
    query = list(train["query"].values)
    product_title = list(train["product_title"].values)
    product_description = list(train["product_description"].values)
    """
    # run the following code to setup test features file
    
    test = pd.read_csv("../data/test.csv", sep = ",").fillna("")
    query = list(test["query"].values)
    product_title = list(test["product_title"].values)
    product_description = list(test["product_description"].values)
    

    for i in range(len(product_description)):  #replace missing description with product title
        if product_description[i] == "" and product_title[i] != "":
            product_description[i] = product_title[i]
        if product_title[i] == "" and product_description[i] != "":
            product_title[i] = product_description[i]
        if product_title[i] == product_description[i] == "":
            product_title[i] = product_description[i] = "a"
        if query[i] == "":
            query[i] = "a"
    df = pd.DataFrame()        
    result1 = score(query, product_title)
    result2 = score(query, product_description)
    df["feature1"] = result1
    df["feature2"] = result2
    #uncomment to extract training features
    #df.to_csv("../data/features.csv", index = False)
    #uncomment to extract test features
    df.to_csv("../data/features_test.csv", index = False)
    

    
       
  