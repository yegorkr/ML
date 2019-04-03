import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
import random
import pandas as pd
from statistics import mean
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def dist(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5
data = pd.read_csv("insurance.csv")
data = data.drop(['region'], 1)
lb = LabelEncoder()
data['sex'] = lb.fit_transform(data['sex'])
data['smoker'] = lb.fit_transform(data['smoker'])
X = data.drop(['charges'], 1)
Y = np.array(data['charges'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

pca = PCA(n_components=6).fit(data)
red_data = PCA(n_components=2).fit_transform(data)
print(red_data)
r1_train, r1_test, r2_train, r2_test = train_test_split(red_data[:, 0:1], Y, test_size = 0.2)

km = KMeans(n_clusters=3, init='k-means++', n_init=10, n_jobs=4)
km.fit(red_data)
h = .2
x_min, x_max = red_data[:, 0].min() - 1, red_data[:, 0].max() + 1
y_min, y_max = red_data[:, 1].min() - 1, red_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
z = km.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
plt.figure(1)
plt.clf()

centroids = km.cluster_centers_

plt.plot(red_data[:, 0], red_data[:, 1], 'k.', markersize = 3)
plt.scatter(centroids[:, 0], centroids[:, 1])
plt.imshow(z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect= 'auto', origin= 'lower')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show() 

c1 = (random.randint(-6000, 42000), random.randint(-15, 20))
c2 = (random.randint(-6000, 42000), random.randint(-15, 20))
c3 = (random.randint(-6000, 42000), random.randint(-15, 20))
cl1 = []
cl2 = []
cl3 = []
for i in range(10):
    cl1 = []
    cl2 = []
    cl3 = []
    for i in red_data[:, :]:
        if(dist(i[0], i[1], c1[0], c1[1]) < dist(i[0], i[1], c2[0], c2[1])):
            if(dist(i[0], i[1], c1[0], c1[1]) < dist(i[0], i[1], c3[0], c3[1])):
                cl1.append(i)
            else: 
                cl3.append(i)
        else: 
            if(dist(i[0], i[1], c2[0], c2[1]) < dist(i[0], i[1], c3[0], c3[1])):
                cl2.append(i)
            else:
                cl3.append(i)
    c1 = (sum(cl1)[0] / cl1.__len__(), sum(cl1)[1] / cl1.__len__())
    c2 = (sum(cl2)[0] / cl2.__len__(), sum(cl2)[1] / cl2.__len__())
    c3 = (sum(cl3)[0] / cl3.__len__(), sum(cl3)[1] / cl3.__len__())
    cl1, cl2, cl3 = np.asarray(cl1), np.asarray(cl2), np.asarray(cl3)
    plt.scatter(cl1[:, 0], cl1[:, 1])
    plt.scatter(cl2[:, 0], cl2[:, 1])
    plt.scatter(cl3[:, 0], cl3[:, 1])
    plt.show()
