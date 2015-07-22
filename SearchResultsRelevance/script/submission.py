# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:17:06 2015

@author: zhihuixie
"""

import math
import pandas as pd
import numpy as np
df1 = pd.read_csv("../data/result.csv")
df2 = pd.read_csv("../data/result1.csv")
df3 = pd.read_csv("../data/result2.csv")
df4 = pd.read_csv("../data/result3.csv")
df5 = pd.read_csv("../data/result4.csv")
df6 = pd.read_csv("../data/result5.csv")
df7 = pd.read_csv("../data/result6.csv")
df = pd.DataFrame()
prediction1  = df1["prediction"].values
prediction2 = df2["prediction"].values 
prediction3 = df3["prediction"].values 
prediction4 = df4["prediction"].values 
prediction5 = df5["prediction"].values 
prediction6 = df6["prediction"].values 
prediction7 = df7["prediction"].values 
predictions = []
for i in range(len(prediction1)):
    x1 = math.pow(prediction1[i] * prediction2[i] * prediction3[i] * prediction4[i]* prediction5[i]* prediction6[i] * prediction7[i], 1.0/7.0)
    x2 = math.exp(math.log(prediction1[i] * prediction2[i] * prediction3[i] * prediction4[i] * prediction5[i]* prediction6[i]* prediction7[i])/7.0)
    x = math.sqrt(x1 * x2)
    x = round(x)
    predictions.append(int(x))  
    
df["id"] = df1.id
df["prediction"] = predictions
df.to_csv("../data/final_result.csv", index = False)