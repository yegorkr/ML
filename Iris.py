import numpy as np
import sklearn
import math
import pandas as pd
from statistics import mean
from sklearn.model_selection import train_test_split

def pV(v, i):
    data1 = data.loc[data[cols[4]] == 'Iris-setosa']
    data2 = data.loc[data[cols[4]] == 'Iris-virginica']
    data3 = data.loc[data[cols[4]] == 'Iris-versicolor']
    v1[i].append(math.exp((-1)*(v-data1[cols[i]].mean(axis = 0))**2/(2*data1[cols[i]].var(axis = 0)))/math.sqrt(2*math.pi*data1[cols[i]].var(axis = 0)))
    v2[i].append(math.exp((-1)*(v-data2[cols[i]].mean(axis = 0))**2/(2*data2[cols[i]].var(axis = 0)))/math.sqrt(2*math.pi*data2[cols[i]].var(axis = 0)))
    v3[i].append(math.exp((-1)*(v-data3[cols[i]].mean(axis = 0))**2/(2*data3[cols[i]].var(axis = 0)))/math.sqrt(2*math.pi*data3[cols[i]].var(axis = 0)))
  
def P(f1, f2, f3, f4):
    i1 = val[0].index(f1)
    i2 = val[1].index(f2)
    i3 = val[2].index(f3)
    i4 = val[3].index(f4)
    P1 = (p1[0][i1] * p1[1][i2] * p1[2][i3] * p1[3][i4] * I1) / (p1[0][i1] * p1[1][i2] * p1[2][i3] * p1[3][i4] * I1 +
    p2[0][i1] * p2[1][i2] * p2[2][i3] * p2[3][i4] * I2 + p3[0][i1] * p3[1][i2] * p3[2][i3] * p3[3][i4] * I3)
    P2 = (p1[0][i1] * p2[1][i2] * p2[2][i3] * p2[3][i4] * I2) / (p1[0][i1] * p1[1][i2] * p1[2][i3] * p1[3][i4] * I1 +
    p2[0][i1] * p2[1][i2] * p2[2][i3] * p2[3][i4] * I2 + p3[0][i1] * p3[1][i2] * p3[2][i3] * p3[3][i4] * I3)
    P3 = (p3[0][i1] * p3[1][i2] * p3[2][i3] * p3[3][i4] * I3) / (p1[0][i1] * p1[1][i2] * p1[2][i3] * p1[3][i4] * I1 +
    p2[0][i1] * p2[1][i2] * p2[2][i3] * p2[3][i4] * I2 + p3[0][i1] * p3[1][i2] * p3[2][i3] * p3[3][i4] * I3)
    if max(P1, P2, P3) == P1: return 'Iris-setosa'
    if max(P1, P2, P3) == P2: return 'Iris-virginica'
    if max(P1, P2, P3) == P3: return 'Iris-versicolor'

def PV(f1, f2, f3, f4):
    i1 = val[0].index(f1)
    i2 = val[1].index(f2)
    i3 = val[2].index(f3)
    i4 = val[3].index(f4)
    P1 = (v1[0][i1] * v1[1][i2] * v1[2][i3] * v1[3][i4] * I1) / (v1[0][i1] * v1[1][i2] * v1[2][i3] * v1[3][i4] * I1 +
    v2[0][i1] * v2[1][i2] * v2[2][i3] * v2[3][i4] * I2 + v3[0][i1] * v3[1][i2] * v3[2][i3] * v3[3][i4] * I3)
    P2 = (v1[0][i1] * v2[1][i2] * v2[2][i3] * v2[3][i4] * I2) / (v1[0][i1] * v1[1][i2] * v1[2][i3] * v1[3][i4] * I1 +
    v2[0][i1] * v2[1][i2] * v2[2][i3] * v2[3][i4] * I2 + v3[0][i1] * v3[1][i2] * v3[2][i3] * v3[3][i4] * I3)
    P3 = (v3[0][i1] * v3[1][i2] * v3[2][i3] * v3[3][i4] * I3) / (v1[0][i1] * v1[1][i2] * v1[2][i3] * v1[3][i4] * I1 +
    v2[0][i1] * v2[1][i2] * v2[2][i3] * v2[3][i4] * I2 + v3[0][i1] * v3[1][i2] * v3[2][i3] * v3[3][i4] * I3)
    if max(P1, P2, P3) == P1: return 'Iris-setosa'
    if max(P1, P2, P3) == P2: return 'Iris-virginica'
    if max(P1, P2, P3) == P3: return 'Iris-versicolor'

data = pd.read_csv("Iris.csv")
data = data.drop(['Id'], 1)
val = [[]]
p1 = [[]]
p2 = [[]]
p3 = [[]]
v1 = [[]]
v2 = [[]]
v3 = [[]]
count = [[]]
cols = data.columns.values
for i, col in enumerate(data.drop(['Species'], 1)):
    val.append([])
    p1.append([])
    p2.append([])
    p3.append([])
    count.append([])
    for j in range(int(data[cols[i]].min() * 10), int(data[cols[i]].max() * 10 + 1)):
        df = data.loc[data[cols[i]] == round(j * 0.1, 3)]
        val[i].append(round(j * 0.1, 3))
        if len(df) == 0: 
            p1[i].append(0)
            p2[i].append(0)
            p3[i].append(0)
            continue
        p1[i].append(len(df.loc[df[cols[4]] == 'Iris-setosa']) / len(df))
        p2[i].append(len(df.loc[df[cols[4]] == 'Iris-virginica']) / len(df))
        p3[i].append(len(df.loc[df[cols[4]] == 'Iris-versicolor']) / len(df))
        count[i].append(len(df))
total = len(data)
for i, col in enumerate(data.drop(['Species'], 1)):
    v1.append([])
    v2.append([])
    v3.append([])
    for j in range(int(data[cols[i]].min() * 10), int(data[cols[i]].max() * 10 + 1)):
        pV(round(j * 0.1, 3), i)
I1 = len(data.loc[data[cols[4]] == 'Iris-setosa']) / total
I2 = len(data.loc[data[cols[4]] == 'Iris-virginica']) / total
I3 = len(data.loc[data[cols[4]] == 'Iris-versicolor']) / total
X = data.drop(['Species'], 1)
Y = np.array(data['Species'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
acc = 0
temp = []
for i in range (len(X_test)):
    if P(X_test.iloc[i][0], X_test.iloc[i][1], X_test.iloc[i][2], X_test.iloc[i][3]) == Y_test[i]: 
        acc += 1
    temp.append(P(X_test.iloc[i][0], X_test.iloc[i][1], X_test.iloc[i][2], X_test.iloc[i][3]))
print(acc / len(Y_test))
df = pd.DataFrame(temp)
df['ans'] = Y_test
accc = 0
ttemp = []
for i in range (len(X_test)):
    if PV(X_test.iloc[i][0], X_test.iloc[i][1], X_test.iloc[i][2], X_test.iloc[i][3]) == Y_test[i]: 
        accc += 1
    ttemp.append(PV(X_test.iloc[i][0], X_test.iloc[i][1], X_test.iloc[i][2], X_test.iloc[i][3]))
print(accc / len(Y_test))
ddf = pd.DataFrame(ttemp)
df['ans1'] = ttemp
print(df)