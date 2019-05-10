# -*- coding: utf-8 -*-


from skimage import data, img_as_float, color
from skimage.feature import corner_harris, corner_peaks
from scipy import ndimage as ndi
from scipy.ndimage.filters import convolve as conv
from scipy.ndimage.filters import gaussian_filter as gauss
import numpy as np
import matplotlib.pyplot as plt


I = data.camera()
I = img_as_float(I)

Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Gy = np.transpose(Gx)
Ix = conv(I, Gx, mode='constant')
Iy = conv(I, Gy, mode='constant')

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(Ix)
plt.subplot(1,2,2)
plt.imshow(Iy)

# Tenseur
Axx = gauss(Ix*Ix, 1, mode='constant')
Ayy = gauss(Iy*Iy, 1, mode='constant')
Axy = gauss(Ix*Iy, 1, mode='constant')

# determinant
detA = Axx * Ayy - Axy ** 2

# trace
traceA = Axx + Ayy

# Response
k = 0.05
R = detA - k * traceA ** 2

#
R = corner_harris(I)
plt.figure(2)
plt.clf()
plt.imshow(R)

# Number of best points
n_top = 100

# Top points from response
R_top = np.isin(R, np.sort(np.ravel(R))[-1:-(n_top+1):-1])

# Suppression of non-maxima
Rmax = ndi.filters.maximum_filter(R, 7)
R_lmax = np.copy(R)
R_lmax[R!=Rmax] = -np.inf
R_int = np.isin(R_lmax, np.sort(np.ravel(R_lmax))[-1:-(n_top+1):-1])

# Display
def draw_int(I, R_int, l=3):
    Irgb = color.gray2rgb(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if R_int[i,j]:                
                Irgb[max(0,i-l):min(I.shape[0],i+l+1),
                     j, :] = [1, 0, 0]
                Irgb[i, 
                     max(0,j-l):min(I.shape[1],j+l+1),
                     :] = [1, 0, 0]
    return Irgb
    

plt.figure(3)
plt.clf()
plt.subplot(1,2,1)
plt.imshow(draw_int(I, R_top))
plt.subplot(1,2,2)
plt.imshow(draw_int(I, R_int))

# 
R = corner_peaks(corner_harris(I))
plt.figure(4)
plt.clf()
plt.subplot(2,3,1)
plt.imshow(I)
plt.plot(R[:,1], R[:,0], 'r+')
for i, s in enumerate(np.arange(1, 6)*0.2):
    Is = gauss(I, s)
    Rs = corner_peaks(corner_harris(Is))
    plt.subplot(2,3,i+2)
    plt.title(f'sigma={s}')
    plt.imshow(Is)
    plt.plot(Rs[:,1], Rs[:,0], 'r+')