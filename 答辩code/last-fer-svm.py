
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.models import Model, Sequential
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import h5py
from matplotlib import pyplot as plt
from PIL import Image
from time import time
import logging#程序进展信息
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split#分割数据集
#from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people#下载数据集
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# In[2]:


emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# In[3]:


num_classes = len(emotion_labels)


# In[4]:


fer_data = h5py.File('./data.h5', 'r', driver='core')


# In[5]:


fer_x = np.asarray(fer_data['Training_pixel']).reshape((28709, 48*48))


# In[6]:


fer_y = fer_data['Training_label']


# In[7]:


X_train = fer_x


# In[8]:


X_test_public=np.asarray(fer_data['PublicTest_pixel']).reshape((3589,48*48))


# In[9]:


X_test_private=np.asarray(fer_data['PrivateTest_pixel']).reshape((3589,48*48))


# In[10]:


num_train = X_train.shape[0]


# In[11]:


num_val_public=X_test_public.shape[0]


# In[12]:


num_val_private=X_test_private.shape[0]


# In[13]:


print('Training: ',X_train.shape)


# In[14]:


print('PublicTest:',X_test_public.shape)


# In[15]:


print('PrivateTest:',X_test_private.shape)


# In[16]:


Y_train = fer_data['Training_label']


# In[17]:


Y_train = to_categorical(Y_train, num_classes)


# In[18]:


Y_test_public=fer_data['PublicTest_label']


# In[19]:


Y_test_private=fer_data['PrivateTest_label']


# In[20]:


Y_test_public=to_categorical(Y_test_public,num_classes)


# In[21]:


Y_test_private=to_categorical(Y_test_private,num_classes)


# In[22]:


print('Training:',Y_train.shape)


# In[23]:


print('PublicTest:',Y_test_public.shape)


# In[24]:


print('PrivateTest:',Y_test_private.shape)


# In[44]:


n_components = 64


# In[45]:


pca =PCA(svd_solver='randomized',n_components=n_components,whiten=True).fit(X_train)#降低维度


# In[46]:


X_train_pca = pca.transform(X_train)      #得到训练集投影系数


# In[47]:


X_val_pca_public = pca.transform(X_test_public) 


# In[48]:


X_val_pca_private = pca.transform(X_test_private) 


# In[49]:


print(X_train_pca.shape)


# In[50]:


print(X_val_pca_public.shape)


# In[51]:


print(X_val_pca_private.shape)


# In[52]:


eigenfaces = pca.components_.reshape((n_components, 48, 48))


# In[53]:


def plot_gallery(images, titles, h, w, n_row=3, n_col=3):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
 
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, 48, 48)
plt.show()


# In[54]:


y_train_ = np.argmax(Y_train, axis=-1)


# In[55]:


y_train_.shape


# In[56]:


print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3],
              'gamma': [0.001], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca[:5000], y_train_[:5000])
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# In[57]:


print("Predicting people's emotions on the test set")
t0 = time()
y_pred_public= clf.predict(X_val_pca_public)
y_pred_private= clf.predict(X_val_pca_private)
print("done in %0.3fs" % (time() - t0))
y_val_public = np.argmax(Y_test_public, axis=-1)
y_val_private = np.argmax(Y_test_private, axis=-1)
print(classification_report(y_val_public, y_pred_public))
print(confusion_matrix(y_val_public, y_pred_public, labels=range(num_classes)))


# In[59]:


print(classification_report(y_val_private, y_pred_private))
print(confusion_matrix(y_val_private, y_pred_private, labels=range(num_classes)))

