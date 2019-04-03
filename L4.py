# -*- coding: utf-8 -*-


import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels


labels = ['Apple Braeburn', 'Avocado', 'Banana', 'Cocos', 'Kiwi', 'Lemon', 'Orange']
labels = {label: i for i, label in enumerate(labels)}
img_size = 100
num_classes = len(labels)


def load_data(folder_path):
    X = []
    y = []
    for root, dirs, files in os.walk(folder_path):
        for basename in files:
            if basename.endswith(".jpg"):
                file_path = os.path.join(root, basename)
                _, label = os.path.split(root)
                label = labels.get(label)
                if label is None:
                    continue
                img = Image.open(file_path)
                img.load()
                img.thumbnail((img_size, img_size))
                img = np.asarray(img, dtype=np.int16).transpose((2, 0, 1))
                X.append(img)
                y.append(label)
    X = np.asarray(X).reshape(-1, 3, img_size, img_size) / 10000
    y = np.asarray(y)
    return X, y


X, y = load_data("C:\\Test\\fruits-360\\Train2")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

X_test, y_test = load_data("C:\\Test\\fruits-360\\Test2")

train_dataset = utils.TensorDataset(torch.Tensor(X_train),torch.LongTensor(y_train))
val_dataset = utils.TensorDataset(torch.Tensor(X_val),torch.LongTensor(y_val))
test_dataset = utils.TensorDataset(torch.Tensor(X_test),torch.LongTensor(y_test))

train_loader = utils.DataLoader(train_dataset,  shuffle=True, batch_size=16)
val_loader = utils.DataLoader(val_dataset, batch_size=16)
test_loader = utils.DataLoader(test_dataset, batch_size=16)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(16928, 64)
        self.fc2 = nn.Linear(64, num_classes)
        print(self.fc2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 16928)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = ConvNet()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(2):  
    print('epoch #{}'.format(epoch))

    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for data in train_loader:
        inputs, targets = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += (outputs.max(dim=1)[1] == targets).sum().item()
    print("train loss {:.4f}".format(running_loss/len(train_loader)))
    print("train acc {:.2f}".format(100*running_acc/len(train_dataset)))

    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    for data in val_loader:
        inputs, targets = data

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        running_acc += (outputs.max(dim=1)[1] == targets).sum().item()
    print("val loss {:.4f}".format(running_loss/len(val_loader)))
    print("val acc {:.2f}".format(100*running_acc/len(val_dataset)))

y_true = []
y_pred = []
with torch.no_grad():
    for data in test_loader:
        images, targets = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred += predicted.tolist()
        y_true += targets.tolist()

class_names = list(labels.keys())
print(classification_report(y_true, y_pred, target_names=class_names))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
plot_confusion_matrix(y_true, y_pred, classes=np.array(class_names), title='Confusion matrix, without normalization')
plot_confusion_matrix(y_true, y_pred, classes=np.array(class_names), normalize=True, title='Normalized confusion matrix')



plt.show()