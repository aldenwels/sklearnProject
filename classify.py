#!/usr/bin/python

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn import model_selection as ms, neighbors, tree
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import KFold

def vectorMachine(chessdata,x,y):
  print('SVM data')
  rbfclf = SVC(kernel='rbf')
  rbfclf.fit(x, y)

  linearclf = SVC(kernel='linear')
  linearclf.fit(x, y)

  sigclf = SVC(kernel='sigmoid')
  sigclf.fit(x, y)

  polyclf = SVC(kernel='poly',degree=3)
  polyclf.fit(x, y)

  print('Rbf svm cross validation: ')
  crossValidation(rbfclf,x,y)
  print('Linear svm cross validation: ')
  crossValidation(linearclf,x,y)
  print('Sigmoid SVM cross validation: ')
  crossValidation(sigclf,x,y)
  print('Polynomial SVM cross validation: ')
  crossValidation(polyclf,x,y)

  #plotSVM(x,y)



def knearest(x,y):
  print('KNN')
  knn = [KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree').fit(x,y),
  KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree').fit(x,y),
  KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree').fit(x,y),
  KNeighborsClassifier(n_neighbors=15, algorithm='kd_tree').fit(x,y)
  ]
  for k in knn:
    print('Cross validation for KNN with nearest neighbors of ' + repr(k.n_neighbors))
    crossValidation(k,x,y)

  #plotKNN(knn,x,y)


def descisionTree(x,y):
  print('decision tree')
  dtrees = [tree.DecisionTreeClassifier(),tree.DecisionTreeClassifier(),
  tree.DecisionTreeClassifier(),tree.DecisionTreeClassifier(),tree.DecisionTreeClassifier()]
  depth = 1
  for t in dtrees:
    t.max_depth = depth
    t.fit(x,y)
    depth = depth * 2
  for s in dtrees:
    print('Cross valid for descision tree with depth '+ repr(s.max_depth))
    crossValidation(s,x,y)
  #plotdtree(dtrees,x,y)

def crossValidation(model,x,y):
  modelScores = ms.cross_val_score(model, x, y, cv = 10)
  print('scores: ' +  repr(modelScores))
  print('standard deviation: '+ repr(np.std(modelScores)))
  mean = 0.0
  for s in modelScores:
    mean += modelScores[s]
  mean = mean / 10
  print('mean: '+ repr(mean))

def plot(labelOne,labelZero):
  plt.plot(labelOne['A'],labelOne['B'],'ro')
  plt.plot(labelZero['A'],labelZero['B'],'go')
  plt.show()

def plotSVM(x,Y):
  h = .02 
  X = np.array(x)
  y = np.array(Y)
  svc = svm.SVC(kernel='sigmoid').fit(X, y)
  rbf_svc = svm.SVC(kernel='rbf').fit(X, y)
  poly_svc = svm.SVC(kernel='poly',degree=12).fit(X, y)
  lin_svc = svm.SVC(kernel='linear').fit(X,y)
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
  # title for the plots
  titles = ['SVC with sigmoid',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial kernel with degree ' + repr(poly_svc.degree)]
  for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
  plt.show()

def plotKNN(KNNs,x,Y):
  print('plot KNN data')
  h = .02 
  X = np.array(x)
  y = np.array(Y)

  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
  # title for the plots
  titles = ['KNN with k at ' + repr(KNNs[0].n_neighbors),
          'KNN with k at ' + repr(KNNs[1].n_neighbors),
          'KNN with k at ' + repr(KNNs[2].n_neighbors),
          'KNN with k at ' + repr(KNNs[3].n_neighbors)]
  for i, knn in enumerate(KNNs):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    print(knn.n_neighbors)
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
  plt.show()

def plotdtree(dtrees,x,Y):
  h = .02 
  X = np.array(x)
  y = np.array(Y)

  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
  # title for the plots
  titles = ['Decision tree with max depth ' + repr(dtrees[0].max_depth),
          'Decision tree with max depth ' + repr(dtrees[1].max_depth),
          'Decision tree with max depth ' + repr(dtrees[2].max_depth),
          'Decision tree with max depth ' + repr(dtrees[3].max_depth)]
  for i, d in enumerate(dtrees):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = d.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
  plt.show()

def main():
    chessdata = pd.read_csv("chessboard.csv")
    x = chessdata[['A','B']]
    y = chessdata['label']
    labelOne = chessdata[chessdata['label']==1]
    labelZero = chessdata[chessdata['label']==0]
    
    #plot(labelOne,labelZero)
    #vectorMachine(chessdata,x,y)
    knearest(x,y)
    descisionTree(x,y)

if __name__ == "__main__":
    main()

