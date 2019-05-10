# -*- coding: utf-8 -*-

from skimage import data , io , util
import numpy as np

im = data.camera()
io.imshow(util.img_as_float(im))

from scipy.ndimage.filters import convolve

Gx= np.array([[-1,0,1]])
Gy=np.transpose(Gx)
I_x =convolve(im, Gx,mode='constant')
I_y =convolve(im, Gy,mode='constant')

img_norme = np.sqrt(I_x**2 + I_y**2)
img_ori = np.arctan2(I_x,I_y)





import numpy as np
from skimage import io, util

img_dim = 24
n_pos = 3000
n_neg = 12000
n_train = n_pos+n_neg

# Charge les images d'apprentissage
img_pos = np.zeros((img_dim, img_dim, n_pos))
img_neg = np.zeros((img_dim, img_dim, n_neg))

# Charge les images positives
for i in range(n_pos):
    im = io.imread("imageface/train/pos/" + "%05d"%(i+1) + ".png")
    img_pos[:,:,i] = util.img_as_float(im)

# Charge les images négatives
for i in range(n_neg):
    im = io.imread("imageface/train/neg/" + "%05d"%(i+1) + ".png")
    img_neg[:,:,i] = util.img_as_float(im)
    
# Images d'apprentissage
img_train = np.concatenate((img_pos, img_neg), axis=2)

# Vecteurs images
x_train = np.zeros((n_train, img_dim*img_dim))
for i in range(n_train):
    x_train[i,:] = np.ravel(img_train[:,:,i])
# Label
y_train = np.concatenate((np.ones(n_pos), -np.ones(n_neg)))

# Training
from sklearn.svm import LinearSVC

clf = LinearSVC()
clf.fit(x_train,y_train)

# Ensemble de test
n_pos_t = 1000
n_neg_t = 5256
n_test = n_pos_t + n_neg_t

# Charge les images de test
img_pos_t = np.zeros((img_dim, img_dim, n_pos_t))
img_neg_t = np.zeros((img_dim, img_dim, n_neg_t))

# Positives
for i in range(n_pos_t):
    im = io.imread("imageface/test/pos/" + "%05d"%(i+1) + ".png")
    img_pos_t[:,:,i] = util.img_as_float(im)

# Négatives
for i in range(n_neg_t):
    im = io.imread("imageface/test/neg/" + "%05d"%(i+1) + ".png")
    img_neg_t[:,:,i] = util.img_as_float(im)

# Images de test
img_test = np.concatenate((img_pos_t, img_neg_t), axis=2)

# Vecteurs images
x_test = np.zeros((n_test, img_dim*img_dim))
for i in range(n_test):
    x_test[i,:] = np.ravel(img_test[:,:,i])
# Labels
y_test = np.concatenate((np.ones(n_pos_t), -np.ones(n_neg_t)))

# Prediction
y_pred = clf.predict(x_test)
print(f"Taux d'erreurs : {np.mean(y_pred != y_test)*100} ")

# Histogram of Oriented Gradient
from skimage.feature import hog
n_dim = 81

# Vecteurs images
x_train_hog = np.zeros((n_train, n_dim))
for i in range(n_train):
    x_train_hog[i,:] = hog(img_train[:,:,i])

# Apprentissage
clf_hog = LinearSVC()
clf_hog.fit(x_train_hog,y_train)

# Vecteurs test
x_test_hog = np.zeros((n_test, n_dim))
for i in range(n_test):
    x_test_hog[i,:] = hog(img_test[:,:,i])
    
# Prediction
y_pred_hog = clf_hog.predict(x_test_hog)
print(f"Taux d'erreurs : {np.mean(y_pred_hog != y_test)*100} ")
