from abc import ABCMeta, abstractmethod, abstractproperty
from itertools import cycle, islice

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import accuracy_score
from sklearn import cluster, datasets, mixture


class ITest(object):
    __metaclass__=ABCMeta

    @abstractmethod
    def __generate_data__(self, size):
        pass

    def __init__(self):
        self.X_train, self.y_train = self.__generate_data__(self.train_size)
        self.X_test, self.y_test = self.__generate_data__(self.test_size)

    def __measure_score__(self, classifier, metrics):
        classifier.fit(self.X_train, self.y_train)
        y_pred = classifier.predict(self.X_test)
        result_scores = dict()
        for metrica_name in metrics:
            result_scores[metrica_name] = metrics[metrica_name](self.y_test, y_pred)

        return y_pred, result_scores

    def __draw_data__(self, X, y_true, y_pred, name):
        plt.subplot(1, 1, 1)
        plt.title(name, size=18)
        error_color = "#ff0000"
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_true) + 1))))

        colors = np.append(colors, ["#000000"])

        colors = colors[y_true]
        actual_colors = np.where(y_true == y_pred, colors, error_color)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=actual_colors)

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def run(self, classifier, metrics={"Accuracy": accuracy_score}, draw=True):
        y_pred, scores = self.__measure_score__(classifier, metrics)
        if draw:
            if len(np.unique(self.y_train)) == 2:
                if self.X_train.shape[1] == 2:
                    self.__draw_data__(
                        self.X_test,
                        self.y_test,
                        y_pred,
                        "Classes distribution"
                    )
        return scores


class TNoisyCirclesTest(ITest):
    def __init__(self, train_size=1000, test_size=100, factor=0.5, noise=0.05):
        self.train_size = train_size
        self.test_size = test_size
        self.factor = factor
        self.noise = noise
        super(TNoisyCirclesTest, self).__init__()

    def __generate_data__(self, size):
        X, y = datasets.make_circles(
            n_samples=size, factor=self.factor, noise=self.noise
        )
        y[y == 0] = -1
        return X, y


if __name__ == "__main__":
    test = TNoisyCirclesTest(test_size=600, noise=0.25)
    classifier = sklearn.svm.SVC()
    test.run(classifier)
    
        