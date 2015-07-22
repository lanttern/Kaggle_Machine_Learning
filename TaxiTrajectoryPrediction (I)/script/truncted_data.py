# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 21:37:20 2015

@author: zhihuixie
"""

import pandas as pd
import numpy as np
import re
df = pd.read_csv("../data/test.csv")
positions = list(df.POLYLINE.values)
LATITUDE, LONGITUDE = [], []
symbol = "["
for position in positions:  
    position = position.replace("[", "")
    position = position.replace("]", "")
    position = position.split(",")
    lon, lat = position[-2], position[-1]
    LATITUDE.append(float(lat)) 
    LONGITUDE.append(float(lon))
df_submission = pd.DataFrame()
df_submission["TRIP_ID"] = df["TRIP_ID"]
df_submission["LATITUDE"] = LATITUDE
df_submission["LONGITUDE"] = LONGITUDE
df_submission.to_csv("../data/result.csv", index = False)



"""
df = df.loc[np.random.choice(df.index, 5, replace = False)]
df.to_csv("../data/sub_train.csv")
"""