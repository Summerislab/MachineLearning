# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/11/23 10:17
# @Author : nxchen
# @File : random_forest.py

import numpy as np
from collections import Counter
from Sample import sample


N_ESTIMATORS = None
MAX_DEPTH = None
SUBSPACE_DIM = None


class Node:
    def __init__(self, dimension, value, y):
        self.dimension = dimension
        self.value = value
        self.y = y
        self.left = None
        self.right = None


class RandomForest(object):
    def __init__(self, n_estimators: int, max_depth: int, subspaces_dim: int, random_state: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subspaces_dim = subspaces_dim
        self.random_state = random_state
        """
          在类构造函数中指定任何必填字段
        """
        self.trees = []

    def fit(self, X, y):
        for i in range(self.n_estimators):
            """
              在您自己的子样本上编写每个算法树的训练函数
            """
            s = sample(X, self.subspaces_dim)
            X_sampled, y_sampled = s(X, y)
            loss = np.inf
            tree = self.create_tree(X_sampled, y_sampled, 0, self.max_depth, loss)
            self.trees.append(tree)

    def create_tree(self, X, y, depth, max_depth, uploss=np.inf):
        if depth == max_depth:
            return None

        # Find a feature and its value to split into two parts
        loss, dim, value = self.split_one_feature(X, y)

        node = Node(dim, value, y)

        # If loss is zero, it's leaf node
        if loss < 0.00001:
            return node

        # if loss > uploss:
        #     return node

        idx_l = X[:, dim] <= value
        idx_r = X[:, dim] > value

        node.left = self.create_tree(X[idx_l], y[idx_l], depth+1, max_depth, uploss=loss)
        node.right = self.create_tree(X[idx_r], y[idx_r], depth+1, max_depth, uploss=loss)

        return node

    def entropy(self, y):
        n = len(y)
        counter = Counter(y)
        p_arr = np.array(list(counter.values())) / n
        entropyLoss = (p_arr * np.log(p_arr) * (-1)).sum()
        return entropyLoss

    def split_one_feature(self, X, y):
        loss = np.inf
        val = None
        n = X.shape[0]
        dim = None
        for feat in range(X.shape[1]):
            feat_uni = np.unique(X[:, feat])
            for idx in range(feat_uni.shape[0]):
                feat_uni_val = feat_uni[idx]
                y_l = X[:, feat] <= feat_uni_val
                p_l = len(X[y_l]) / n
                y_r = X[:, feat] > feat_uni_val
                p_r = len(X[y_r]) / n
                my_loss = p_l * self.entropy(y[y_l]) + p_r * self.entropy(y[y_r])
                if my_loss < loss:
                    loss = my_loss
                    val = feat_uni_val
                    dim = feat
                if loss == 0:
                    break
        return loss, dim, val

    def predict(self, X_test):
        """
          Напишите функцию получения среднего предсказания алгоритма
          编写函数以获取算法的平均预测
        """
        def travel(node, X_test):
            if X_test[node.dimension] <= node.value and node.left:
                y = travel(node.left, X_test)
            elif X_test[node.dimension] > node.value and node.right:
                y = travel(node.right, X_test)
            else:
                counter = Counter(node.y)
                y = counter.most_common(1)[0][0]
            return y

        preds = []
        for i in range(self.n_estimators):
            y_pred = []
            tree = self.trees[i]
            for x in X_test:
                y_pred.append(travel(tree, x))
            preds.append(y_pred)

        # (n_estimators, len(X_test))
        preds = np.array(preds)

        y_vote = [Counter(np.array(preds)[:, i]).most_common(1)[0][0] for i in range(X_test.shape[0])]

        return y_vote
