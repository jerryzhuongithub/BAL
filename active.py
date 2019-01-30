import sys
import numpy as np
import pandas as pd
import time
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from logistic import LogisticRegression
import matplotlib.pyplot as plt

sys.path.append('.')


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


class Sampler(object):
    def sampling_strategy(self, clf, X, j):
        raise Exception("Please implement")

    def reset(self):
        raise Exception("Please implement")


class MarginSampler(Sampler):
    def __init__(self):
        self.schedule = None

    def sampling_strategy(self, clf, X, j):
        n = np.shape(X)[0]
        if self.schedule is None:
            self.schedule = np.arange(n)
        if clf.coef_ is None:
            np.random.shuffle(self.schedule)
            idxs = self.schedule[:j]
            self.schedule = self.schedule[j:]
            return idxs.tolist()
        pk = clf.predict_proba(X[self.schedule])
        pk.sort(axis = 1)
        margin = pk[:,-1] - pk[:,-2]
        top = self.schedule[np.argsort(margin)]
        lmargin = np.repeat(np.NaN, n)
        lmargin[self.schedule] = margin
        idxs = top[:j]
        self.schedule = np.setdiff1d(self.schedule, idxs)
        return idxs[::-1].tolist()

    def reset(self):
        self.schedule = None


class EntropySampler(Sampler):
    def __init__(self):
        self.schedule = None

    def sampling_strategy(self, clf, X, j):
        n = np.shape(X)[0]
        if self.schedule is None:
            self.schedule = np.arange(n)
        if clf.coef_ is None:
            np.random.shuffle(self.schedule)
            idxs = self.schedule[:j]
            self.schedule = self.schedule[j:]
            return idxs.tolist()
        pk = clf.predict_proba(X[self.schedule])
        s = np.sum(pk * np.log(pk), axis = 1)
        top = self.schedule[np.argsort(s)]
        ls = np.repeat(np.NaN, n)
        ls[self.schedule] = s
        idxs = top[:j]
        self.schedule = np.setdiff1d(self.schedule, idxs)
        return idxs.tolist()

    def reset(self):
        self.schedule = None


def training(sampler, k, x_train, y_train, x_test, y_test, j, N, trials, showprogress=True, regularizer=1.0):
    test_err = {}
    for t in range(trials):
        idxs = []
        clf = LogisticRegression(C=regularizer, k=k)
        sampler.reset()
        while len(idxs) < N:
            start_time = time.time()
            if growth == 'geometric':
                idxs += sampler.sampling_strategy(clf, x_train, max(len(idxs), j))
            else:
                idxs += sampler.sampling_strategy(clf, x_train, j)
            clf.fit(x_train[idxs,:], y_train[idxs])
            predicted = clf.predict(x_test)
            err = 1 - np.mean(predicted == y_test)

            if t == 0:
                test_err[len(idxs)] = [err]
            else:
                test_err[len(idxs)].append(err)

            elapsed_time = time.time() - start_time
            if showprogress:
                print t, len(idxs), err,  "%.2fs" % elapsed_time

    return test_err


# Maximum training set size. Training set size is incremented upto this maximum value.
N = option('N', type=int, default=600)
# Rate of growth of training set size
j = option('j', type=int, default=10)
# Training set size can be incremented using geometric or linear progression.
growth = option('growth', choices=['linear', 'geometric'], default='linear')
# Total number of trials to run this experiment for.
trials = option('trials', type=int, default=20)
# Maximum entropy or smallest margin can be selected as the sampling strategy
active_learning_sampling_strategy = option('strategy', choices=['margin', 'entropy'], default='entropy')
# List of newsgroups categories to be considered for classification task.
# Example :- "rec.sport.hockey,rec.sport.baseball,sci.electronics".
# If left unspecified, all the 20 categories are included in task.
categories = option('categories', default=None)
# Filename to store the err plot
err_filename = option('err_filename', default='err.png')


if active_learning_sampling_strategy == 'margin':
    active_learning_sampling_strategy = MarginSampler()
elif active_learning_sampling_strategy == 'entropy':
    active_learning_sampling_strategy = EntropySampler()

if categories is None:
    categories = [1] * 20
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

test_err = training(active_learning_sampling_strategy, len(categories), x_train, y_train, x_test, y_test, j, N, trials)

# Min-max-median plot for test error
task_name = os.path.splitext(err_filename)[0]
test_err = pd.DataFrame.from_dict(test_err)
df = pd.DataFrame({'active': test_err.median()})
ax = df.plot(title=task_name)
ax.fill_between(df.index, test_err.min(), test_err.max(), facecolor='C0', alpha=0.5)
ax.set(xlabel='Training Set Size', ylabel='Test Error')
plt.savefig('figs/active/'+err_filename, dpi=300)
plt.close()
