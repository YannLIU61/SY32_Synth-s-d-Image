# -*- coding: utf-8 -*-


import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from skimage.feature import daisy

def d_daisy(im):
    """
    Calcul des caractéristiques DAISY sous la forme d'un tableau
    """
    descs = daisy(im, step=16, radius=16, rings=3, histograms=8,
                  orientations=8, visualize=False)
    return np.reshape(descs,
                      (descs.shape[0]*descs.shape[1], descs.shape[2]),
                      order='F')
    
# Calcul des caractéristiques DAISY
descr_num = np.zeros(300)
descr = np.empty((0,200))
for i in range(300):
    print(i)
    im = rgb2gray(plt.imread('images/%03d.jpg'%(i)))
    descr = np.concatenate((descr, d_daisy(im)))
    descr_num[i] = len(descr)

# Clustering pour construire le dictionnaire de mots visuels
from sklearn.cluster import KMeans
n_clust = 20
kmeans = KMeans(n_clusters=n_clust)
kmeans.fit(descr)

# Description des images sous la forme d'un histogramme sur le dictionnaire
label = kmeans.predict(descr)
descr_hist = np.zeros((300,n_clust))
for i in range(300):
    print(i)
    i_start = int(descr_num[i-1]) if i > 0 else 0
    i_end = int(descr_num[i])
# 我们可以看到x中最大的数为7，因此bin的数量为8，那么它的索引值为0->7
#x = np.array([0, 1, 1, 3, 2, 1, 7])
# 索引0出现了1次，索引1出现了3次......索引5出现了0次......
#
#因此，输出结果为：array([1, 3, 1, 1, 0, 0, 0, 1])

    descr_hist[i,:] = np.bincount(label[i_start:i_end], minlength=n_clust)
    descr_hist[i,:] /= np.sum(descr_hist[i,:])

# Recherche des images les plus proches
img = ['crab.jpg', 'kangaroo.jpg', 'cougar.jpg']
for i in range(len(img)):
    im0 = rgb2gray(plt.imread(img[i]))
    descr0 = d_daisy(im0)
    label0 = kmeans.predict(descr0)
    descr_hist0 = np.array([np.bincount(label0, minlength=n_clust)]).astype(float)
    descr_hist0 /= np.sum(descr_hist0)
    
    d = distance_matrix(descr_hist0, descr_hist)
    d_i = np.argsort(d[0,:])
    
    plt.figure(img[i])
    plt.clf()
    plt.subplot(3,3,1)
    plt.imshow(plt.imread(img[i]))
    for i in range(2, 9+1):
        plt.subplot(3,3,i)
        plt.imshow(plt.imread('images/' + '%03d'%(d_i[i-2]) + '.jpg'))
        