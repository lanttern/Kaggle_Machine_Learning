# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:16:07 2015

@author: zhihuixie
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from load_data import feature_label_split
    
features_train, labels_train = feature_label_split("../data/train.csv")

pca_components = [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 120, 150, 180, 200]
pca_fits = []

for comp in pca_components:
    pca_fits.append(decomposition.PCA(n_components=comp).fit(features_train))

figure = plt.figure()

d = np.array(labels_train)
# choose number to show
numbers = []
numbers.append(np.argwhere(d == 0)[-3])
numbers.append(np.argwhere(d == 1)[-3])
numbers.append(np.argwhere(d == 2)[-3])
numbers.append(np.argwhere(d == 3)[-3])
numbers.append(np.argwhere(d == 4)[-3])
numbers.append(np.argwhere(d == 5)[-3])
numbers.append(np.argwhere(d == 6)[-3])
numbers.append(np.argwhere(d == 7)[-3])
numbers.append(np.argwhere(d == 8)[-3])
numbers.append(np.argwhere(d == 9)[-3])

#print np.argwhere(d == 0)

pca_index = 1
for n in numbers:
    # plot figures for each number
    for p in pca_fits:
        transformed = p.transform(features_train[n])
        reconstructed = p.inverse_transform(transformed)
        f = figure.add_subplot(10, len(pca_components), pca_index).\
           imshow(np.reshape(reconstructed, (28, 28)), interpolation="nearest",\
           cmap = plt.get_cmap("gray"), vmin = 0, vmax = 255)
        for xlabel in f.axes.get_xticklabels():
            xlabel.set_visible(False)
            xlabel.set_fontsize(0.0)
        for xlabel in f.axes.get_yticklabels():
            xlabel.set_fontsize(0.0)
            xlabel.set_visible(False)
        for tick in f.axes.get_xticklines():
            tick.set_visible(False)
        for tick in f.axes.get_yticklines():
            tick.set_visible(False)
        pca_index += 1

plt.show()
figure.savefig("../result/figure.jpg", transparent=True) # save figure