from __future__ import print_function
import sys
import os
import numpy as np
import pandas as pd
import time
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


class options(object):
    # Rate of growth of training set size in phase 1 - until stopping condition met
    phase1_increment = option('phase1-increment', type=int, default=1)
    # Rate of growth of training set size in phase 2
    phase2_increment = option('phase2-increment', type=int, default=1)
    # Regularization value for Logistic Regression
    regularizer = option('regularizer', type=float, default=1.0)
    # Training set size can be incremented using geometric or linear progression.
    growth = option('growth', choices=['linear', 'geometric'], default='linear')
    # Total number of trials to run this experiment for.
    trials = option('trials', type=int, default=20)
    # Number of training items to be selected can be a fixed value, klogk (where k is the number of classes),
    # until we have at least one training item corresponding to each of the k labels, or until we have m
    # items from each k labels.
    phase1 = option('phase1', choices=['klogk','fixed','until-all-labels','m-per-class'], default='until-all-labels')
    # Sampling strategy in phase 2 can be passive, maximum entropy or smallest margin
    phase2 = option('phase2', choices=['passive', 'margin', 'entropy'], default='passive')
    # Number of items to be selected in phase 1 if we want to have a fixed number of items.
    fixed = option('fixed', type=int, default=1)
    # Number of items m from each of the k classes.
    m_per_class = option('m-per-class', type=int, default=1)
    show_progress = True


class UntilAllLabelsStopping():
    # TODO Generalize to m examples per class
    def __init__(self, y):
        self.y = y
        self.k = len(np.unique(y))

    def met(self, idxs):
        labels = len(np.unique(self.y[idxs]))
        return labels == self.k


class FixedStopping():
    def met(self, idxs):
        if options.fixed is None:
            raise Exception("Requires option --fixed [n]")
        return len(idxs) >= options.fixed


class KLogKStopping():
    def __init__(self, y):
        k = len(np.unique(y))
        self.j = int(k * np.log(k))

    def met(self, idxs):
        return len(idxs) >= self.j


class TrivialStopping():
    def met(self, idxs):
        return True


class PerClass:
    def __init__(self, idxs, x_train, y_train, m_per_class):
        self.schedule = np.arange(np.shape(x_train)[0])
        self.schedule = np.setdiff1d(self.schedule, idxs)
        self.classes = np.unique(y_train)
        self.y_train = y_train
        self.m_per_class = m_per_class

    def sample(self, clf):
        idxs = []
        for k in self.classes:
            yk = np.where(self.y_train[self.schedule] == k)[0]
            np.random.shuffle(yk)
            idxs += yk[:self.m_per_class].tolist()
        self.schedule = np.setdiff1d(self.schedule, idxs)
        return idxs


class Passive:
    def __init__(self, idxs, x_train, increment):
        self.schedule = np.arange(np.shape(x_train)[0])
        self.schedule = np.setdiff1d(self.schedule, idxs)
        np.random.shuffle(self.schedule)
        self.increment = increment

    def sample(self, clf):
        idxs = self.schedule[:self.increment]
        self.schedule = self.schedule[self.increment:]
        return idxs.tolist()


class Margin:
    def __init__(self, idxs, x_train, increment):
        self.schedule = np.arange(np.shape(x_train)[0])
        self.schedule = np.setdiff1d(self.schedule, idxs)
        np.random.shuffle(self.schedule)
        self.increment = increment
        self.x_train = x_train
        self.n = np.shape(x_train)[0]

    def sample(self, clf):
        pk = clf.predict_proba(self.x_train[self.schedule])
        pk.sort(axis = 1)
        margin = pk[:,-1] - pk[:,-2]
        top = self.schedule[np.argsort(margin)]
        lmargin = np.repeat(np.NaN, self.n)
        lmargin[self.schedule] = margin
        idxs = top[:self.increment]
        self.schedule = np.setdiff1d(self.schedule, idxs)
        return idxs[::-1].tolist()


class Entropy:
    def __init__(self, idxs, x_train, increment):
        self.schedule = np.arange(np.shape(x_train)[0])
        self.schedule = np.setdiff1d(self.schedule, idxs)
        np.random.shuffle(self.schedule)
        self.increment = increment
        self.x_train = x_train
        self.n = np.shape(x_train)[0]

    def sample(self, clf):
        pk = clf.predict_proba(self.x_train[self.schedule])
        s = np.sum(pk * np.log(pk), axis = 1)
        top = self.schedule[np.argsort(s)]
        ls = np.repeat(np.NaN, self.n)
        ls[self.schedule] = s
        idxs = top[:self.increment]
        self.schedule = np.setdiff1d(self.schedule, idxs)
        return idxs.tolist()


def get_phase1(idxs, x_train, y_train):
    if options.phase1 == 'fixed':
        return Passive(idxs, x_train, options.phase1_increment)
    elif options.phase1 == 'klogk':
        return Passive(idxs, x_train, options.phase1_increment)
    elif options.phase1 == 'until-all-labels':
        return Passive(idxs, x_train, options.phase1_increment)
    elif options.phase1 == 'm-per-class':
        return PerClass(idxs, x_train, y_train, options.m_per_class)
    else:
        raise Exception("%s not recognized " % options.phase1)


def get_phase2(idxs, x_train):
    if options.phase2 == 'passive':
        return Passive(idxs, x_train, options.phase2_increment)
    elif options.phase2 == 'margin':
        return Margin(idxs, x_train, options.phase2_increment)
    elif options.phase2 == 'entropy':
        return Entropy(idxs, x_train, options.phase2_increment)
    else:
        raise Exception("%s not recognized " % options.phase2)


def get_stopping_condition(y_train):
    if options.phase1 == 'fixed':
        return FixedStopping()
    elif options.phase1 == 'klogk':
        return KLogKStopping()
    elif options.phase1 == 'm-per-class':
        return TrivialStopping()
    elif options.phase1 == 'until-all-labels':
        return UntilAllLabelsStopping(y_train)


def training(k, x_train, y_train, x_test, y_test):
    test_err = {}
    stopped_phase1_at = []
    for t in range(options.trials):
        idxs = []
        clf = LogisticRegression(C=options.regularizer, k=k)
        phase1 = get_phase1(idxs, x_train, y_train)
        stopping_condition = get_stopping_condition(y_train)
        stopped = False
        while len(idxs) < options.N:
            start_time = time.time()
            if not stopped:
                idxs += phase1.sample(clf)
                if stopping_condition.met(idxs):
                    stopped_phase1_at.append(len(idxs))
                    phase2 = get_phase2(idxs, x_train)
                    stopped = True
            else:
                idxs += phase2.sample(clf)
            clf.fit(x_train[idxs,:], y_train[idxs])
            predicted = clf.predict(x_test)
            err = 1 - np.mean(predicted == y_test)

            if t == 0:
                test_err[len(idxs)] = [err]
            else:
                test_err[len(idxs)].append(err)

            elapsed_time = time.time() - start_time
            if options.show_progress:
                print(t, len(idxs), err, clf.funcalls_lastfit, "%.2fs" % elapsed_time)

    return test_err, stopped_phase1_at


def plot_minmedianmax(test_err, stopped_phase1_at, title=None, folder=None, filename=None, save=False):
    test_err = pd.DataFrame.from_dict(test_err)
    df = pd.DataFrame({'active': test_err.median()})
    ax = df.plot(title=title or folder)
    stopped_at = [np.median(stopped_phase1_at)]
    ax.fill_between(df.index, test_err.min(), test_err.max(), facecolor='C0', alpha=0.5)
    ax.axvline(x=stopped_at, color='C', linestyle=':')
    ax.set(xlabel='Training Set Size', ylabel='Test Error')
    if save:
        f = filename or "minmedianmax.png"
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(os.path.join(folder, f), dpi=150)
        plt.close()