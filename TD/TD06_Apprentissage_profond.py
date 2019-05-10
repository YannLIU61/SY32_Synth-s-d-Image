# -*- coding: utf-8 -*-


from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')
print(model.summary())

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
img = load_img('crab.jpg', target_size=(224, 224))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

y = model.predict(img)

from keras.applications.vgg16 import decode_predictions
label = decode_predictions(y)

from keras.models import Model
model_feat = Model(inputs=model.input, 
                   outputs=model.get_layer('fc2').output)
x = model_feat.predict(img)

descr = np.empty((300,4096))
for i in range(300):
    print(i)
    img = load_img('images/%03d.jpg'%(i), target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    descr[i,:] = model_feat.predict(img)
    
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
image = ['crab.jpg', 'kangaroo.jpg', 'cougar.jpg']
for i in range(len(image)):
    img = load_img(image[i], target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    descr0 = model_feat.predict(img)
    
    d = distance_matrix(descr0, descr)
    d_i = np.argsort(d[0,:])
    
    plt.figure(image[i])
    plt.clf()
    plt.subplot(3,3,1)
    plt.imshow(plt.imread(image[i]))
    for i in range(2, 9+1):
        plt.subplot(3,3,i)
        plt.imshow(plt.imread('images/' + '%03d'%(d_i[i-2]) + '.jpg'))
