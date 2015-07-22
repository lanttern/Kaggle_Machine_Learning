# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:45:05 2015

@author: zhihuixie
"""
import pandas as pd

def relevance(query, product, gap):
    """
    function to calculate similarity score for query and product using DP algorithm
    """
    from sklearn.preprocessing import MinMaxScaler
    similarity_score = []
    for index in range(len(query)):
        query_length = len(query[index])
        product_length = len(product[index])
        s = [[0]*(query_length + 1) for i in range(product_length + 1)]
        best_score = 0
        for i in range(1, product_length  + 1):
            for j in range(1, query_length + 1):
                if query[index][j-1] == product[index][i-1]:
                   s[i][j] =max(s[i-1][j] + gap, s[i][j-1] + gap, s[i-1][j-1] + 1, 0)
                else:
                   s[i][j] =max(s[i-1][j] + gap, s[i][j-1] + gap, s[i-1][j-1], 0) 
                if s[i][j] > best_score:
                     best_score = s[i][j]
        similarity_score.append(best_score*1.0)
    scale = MinMaxScaler()
    scores = scale.fit_transform(similarity_score)
    return scores
    
def count_word(query, product):
    from sklearn.preprocessing import MinMaxScaler
    word_freq = []
    for index in range(len(query)):
        query_words = query[index].split(" ")
        counts = 0
        product_words = product[index].split(" ")
        visted = set([])
        for word in query_words:
            word_upper = word[0].upper() + word[1:]
            word_upper_all = word.upper()
            words = [word, word_upper, word_upper_all]
            for w in product_words:
                if w in words and w not in visted:
                    visted.add(w)
                    counts += 1
        #print query[index], counts, len(product_words)
        word_freq.append(counts*1.0/len(query_words))
    scale = MinMaxScaler()
    freq = scale.fit_transform(word_freq)
    return freq


if __name__ == "__main__":
    # run the following code to setup training features file
    """
    train = pd.read_csv("../data/train.csv", sep = ",").fillna("")
    query = list(train["query"].values)
    product_title = list(train["product_title"].values)
    product_description = list(train["product_description"].values)
    
    # run the following code to setup test features file
    """
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
            
    #uncomment to extract training features
    #df = pd.read_csv("../data/features.csv")  
    #uncomment to extract test features
    df = pd.read_csv("../data/features_test.csv")  
    
    result1 = relevance(query, product_title, -5)
    result2 = relevance(query, product_description, -5)
    result3 = count_word(query, product_title)
    result4 = count_word(query, product_description)
    df["feature3"] = result1
    df["feature4"] = result2
    df["feature5"] = result3
    df["feature6"] = result4
    
    #uncomment to extract training features
    #df.to_csv("../data/features.csv", index = False)
    #uncomment to extract test features
    df.to_csv("../data/features_test.csv", index = False)