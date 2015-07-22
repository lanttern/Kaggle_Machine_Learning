# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:26:46 2015

@author: zhihuixie
"""
from sklearn.feature_extraction import text
import pandas as pd
#train = pd.read_csv("../input/train.csv").fillna("")
#test  = pd.read_csv("../input/test.csv").fillna("")

sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
print len(stop_words)
for stw in stop_words:
    sw.append("q"+stw)
    sw.append("z"+stw)
stop_words = text.ENGLISH_STOP_WORDS.union(sw)

print len(stop_words)

def replace_stop_words(list1):
    for s in list1:
        