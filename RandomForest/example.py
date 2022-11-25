# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/11/23 10:55
# @Author : nxchen
# @File : example.py

from sklearn import datasets
from sklearn.model_selection import train_test_split
from random_forest import RandomForest
import numpy as np
from Node import Node, DTree

def main():
    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=True, random_state=42)

    dim = X.shape[1]
    clf = RandomForest(n_estimators=1, max_depth=5, subspaces_dim=dim, random_state=42)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    score = sum(np.array(y_preds == y_test)) / y_test.shape[0]
    print(score)

    dt = DTree()
    dt.fit(X_train, y_train)
    y_preds = dt.predict(X_test)
    print(sum(np.array(y_preds == y_test)) / y_test.shape[0])


main()
