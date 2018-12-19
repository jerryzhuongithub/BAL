import sys
import numpy as np
import pandas as pd
import time
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


class LearnerWrapper(object):
    def __init__(self, clf):
        self.clf = clf
        self.proxy = None

    def fit(self, X,y):
        if np.all(y == y[0]):
            self.proxy = y[0]
        else:
            self.clf.fit(X,y)

    def predict(self, X):
        if self.proxy is None:
            return self.clf.predict(X)
        else:
            return self.proxy

    def __getattr__(self, name):
        if name == "C_":
            if self.proxy is not None:
                return [None]
            else:
                return self.clf.C_

def option(name, type=None, default=None, choices=None):
    f = "--" + name
    for i, arg in enumerate(sys.argv):
        if arg == f or arg == f[1:]:
            v = sys.argv[i+1]
            if type is not None:
                v = type(v)
            if choices and v not in choices:
                raise Exception("Valid choices are [" + str.join(",", map(str, choices)) + "]")
            return v
    return default

# Maximum training set size. Training set size is increamented upto this maximum value.
N = option('N', type=int, default=500)
# Rate of growth of training set size
j = option('j', type=int, default=10)
# Number of folds for cross-validation.
cv = option('cv', type=int, default=5)
# Training set size can be increamented using geometric or linear progression.
growth = option('growth',  choices=['linear', 'geometric'], default='linear')
# Total number of trials to run this experiment for.
trials = option('trials', type=int, default=10) 
# List of newsgroups categories to be considered for classification task.
# Example :- "rec.sport.hockey,rec.sport.baseball,sci.electronics".
# If left unspecified, all the 20 categories are included in task.
categories = option('categories', default=None)
# Filename to store the err plot
err_filename = option('err_filename', default='err.png')

if categories is None:
    corpus = fetch_20newsgroups(subset='all')
else:
    categories = str.split(categories, ",")
    corpus = fetch_20newsgroups(subset='all', categories=categories)

vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(corpus.data)
y = corpus.target

indices = np.arange(len(y))
x_train, x_rest, y_train, y_rest, idx_train, idx_rest = train_test_split(X, y, indices, test_size=0.67)
x_teach, x_test, y_teach, y_test, idx_teach, idx_test = train_test_split(x_rest, y_rest, idx_rest, test_size=0.5)

print len(y_train), "training items"

# Stores test error for varying training set size for all trials.
test_err = {}
for t in range(trials):
    idxs = []
    while len(idxs) < N:
        start_time = time.time()
        if growth == 'geometric':
            idxs += np.random.choice(len(y_train), max(len(idxs),j)).tolist()
        else:
            idxs += np.random.choice(len(y_train), j).tolist()
        kf = KFold(n_splits=min(cv,len(idxs)))
        clf = LearnerWrapper(LogisticRegressionCV(cv=kf))
        clf.fit(x_train[idxs,:], y_train[idxs])
        predicted = clf.predict(x_test)
        err = 1 - np.mean(predicted == y_test)

        if t==0:
            test_err[len(idxs)] = [err]
        else:
            test_err[len((idxs))].append(err)

        elapsed_time = time.time() - start_time
        print t, len(idxs), err,  "%.2fs" % elapsed_time

task_name = os.path.splitext(err_filename)[0]
ax = pd.DataFrame(data=test_err).boxplot()
ax.set(xlabel='Training Set Size', ylabel='Test Error', title = task_name)
plt.savefig('figs/'+err_filename)
