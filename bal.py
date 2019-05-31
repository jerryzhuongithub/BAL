from __future__ import print_function
import sys
sys.path.append('.')
import logging
import numpy as np
import pandas as pd
import time
import copy
import tqdm
import xlsxwriter
import csv
import urllib.request
import os
import zipfile
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from logistic import LogisticRegression

def default_options():
    opts = Options()
    opts.option('phase1-increment', type=int, default=1)
    opts.option('phase2-increment', type=int, default=1)
    opts.option('regularizer', type=float, default=1.0)
    opts.option('trials', type=int, default=20)
    opts.option('phase1', choices=['klogk','fixed','until-all-labels','m-per-class','skip'], default = 'fixed')
    opts.option('phase2', choices=['passive', 'margin', 'entropy'], default = 'passive')
    opts.option('fixed', type=int, default=1)
    opts.option('m-per-class', type=int, default=1)
    opts.option('minipool-size', type=int, default=10)
    return opts

# log = Log(option('log', default=None))

class UntilAllLabelsStopping():
    # TODO Generalize to m examples per class, use same m-per-class flag
    def __init__(self, y):
        self.y = y
        self.k = len(np.unique(y))

    def met(self, idxs):
        labels = len(np.unique(self.y[idxs]))
        return labels == self.k

class AlwaysAlreadyStopped():
    def met(self, idxs):
        return True

class FixedStopping():
    def __init__(self, fixed):
        self.fixed = fixed

    def met(self, idxs):
        if self.fixed is None:
            raise Exception("Requires option --fixed [n]")
        return len(idxs) >= self.fixed

class KLogKStopping():
    def __init__(self, y):
        k = len(np.unique(y))
        self.j = int(k * np.log(k))

    def met(self, idxs):
        return len(idxs) >= self.j

class PerClass:
    def __init__(self, x_train, y_train, m_per_class, increment):
        self.classes = np.unique(y_train)
        self.y_train = y_train
        self.m_per_class = m_per_class
        self.increment = increment
        self.called = False

    def sample(self, clf, log, t):
        if not self.called:
            idxs = []
            for k in self.classes:
                yk = np.where(self.y_train == k)[0]
                np.random.shuffle(yk)
                idxs += yk[:self.m_per_class].tolist()
            self.buffer = idxs
            self.called = True
        idxs = self.buffer[:self.increment]
        self.buffer = self.buffer[self.increment:]
        return idxs

class MinipoolB:
    def __init__(self, b, idxs, x_train, y_train, x_teach, y_teach, increment):
        self.b = b
        self.schedule = np.arange(np.shape(x_train)[0])
        self.schedule = np.setdiff1d(self.schedule, idxs)
        np.random.shuffle(self.schedule)
        self.increment = increment
        self.idxs = copy.deepcopy(idxs)
        self.x_train = x_train
        self.y_train = y_train
        self.x_teach = x_teach
        self.y_teach = y_teach

    def sample(self, clf, log, t):
        candidates = np.random.choice(self.schedule, min(len(self.schedule), self.b), replace=False)
        newclf = copy.deepcopy(clf)
        scores = []
        for c in candidates:
            z = self.idxs + [c]
            newclf.fit(self.x_train[z], self.y_train[z])
            predicted = newclf.predict(self.x_teach)
            err = 1 - np.mean(predicted == self.y_teach)
            scores.append(err)
        best = candidates[np.argsort(scores)][:self.increment]
        self.schedule = np.setdiff1d(self.schedule, best)
        self.idxs.extend(best)
        log.minipoolb.candidates.info(t, candidates)
        log.minipoolb.validation_err.info(t, scores)
        return best.tolist()

class Passive:
    def __init__(self, idxs, x_train, increment):
        self.schedule = np.arange(np.shape(x_train)[0])
        self.schedule = np.setdiff1d(self.schedule, idxs)
        np.random.shuffle(self.schedule)
        self.increment = increment

    def sample(self, clf, log, t):
        idxs = self.schedule[:self.increment]
        self.schedule = self.schedule[self.increment:]
        return idxs.tolist()

class Margin:
    def __init__(self, idxs, x_train, increment):
        self.schedule = np.arange(np.shape(x_train)[0])
        self.schedule = np.setdiff1d(self.schedule, idxs)
        # We sort so that there is no particular logic to ties
        np.random.shuffle(self.schedule)
        self.increment = increment
        self.x_train = x_train
        self.n = np.shape(x_train)[0]

    def sample(self, clf, log, t):
        pk = clf.predict_proba(self.x_train[self.schedule])
        pk.sort(axis = 1)
        margin = pk[:,-1] - pk[:,-2]
        top = self.schedule[np.argsort(margin)]
        lmargin = np.repeat(np.NaN, self.n)
        lmargin[self.schedule] = margin
        n = self.n - len(self.schedule)
        log.margins.debug(n, lmargin)
        log.sorted_margins.debug(n, np.sort(margin))
        log.top_margins.debug(n, top)
        idxs = top[:self.increment]
        self.schedule = np.setdiff1d(self.schedule, idxs)
        return idxs.tolist()

class Entropy:
    def __init__(self, idxs, x_train, increment):
        self.schedule = np.arange(np.shape(x_train)[0])
        self.schedule = np.setdiff1d(self.schedule, idxs)
        np.random.shuffle(self.schedule)
        self.increment = increment
        self.x_train = x_train
        self.n = np.shape(x_train)[0]

    def sample(self, clf, log, t):
        pk = clf.predict_proba(self.x_train[self.schedule])
        s = np.sum(pk * np.log(pk), axis = 1)
        top = self.schedule[np.argsort(s)]
        ls = np.repeat(np.NaN, self.n)
        ls[self.schedule] = s
        n = self.n - len(self.schedule)
        log.entropies.debug(n, ls)
        log.sorted_entropies.debug(n, np.sort(s))
        log.top_entropies.debug(n, top)
        # s computes the negative entropy so taking smallest is appropriate
        idxs = top[:self.increment]
        self.schedule = np.setdiff1d(self.schedule, idxs)
        return idxs.tolist()

class GreedyAccuracy:
    def __init__(self, idxs, x_train, y_train, x_teach, y_teach, increment):
        self.schedule = np.arange(np.shape(x_train)[0])
        self.schedule = np.setdiff1d(self.schedule, idxs)
        np.random.shuffle(self.schedule)
        self.increment = increment
        self.idxs = copy.deepcopy(idxs)
        self.x_train = x_train
        self.y_train = y_train
        self.x_teach = x_teach
        self.y_teach = y_teach

    def sample(self, clf, log, t):
        newclf = copy.deepcopy(clf)
        scores = []
        for c in self.schedule:
            z = self.idxs + [c]
            newclf.fit(self.x_train[z], self.y_train[z])
            predicted = newclf.predict(self.x_teach)
            err = 1 - np.mean(predicted == self.y_teach)
            scores.append(err)
        best = self.schedule[np.argsort(scores)][:self.increment]
        self.schedule = np.setdiff1d(self.schedule, best)
        self.idxs.extend(best)
        return best.tolist()

class GreedyCoverage:
    def __init__(self, idxs, x_train, y_train, increment, feature_offsets, class_proportions, feature_trials):
        self.schedule = np.arange(np.shape(x_train)[0])
        self.schedule = np.setdiff1d(self.schedule, idxs)
        # We sort so that there is no particular logic to ties
        np.random.shuffle(self.schedule)
        self.increment = increment
        self.x_train = x_train
        self.y_train = y_train
        self.x_features = [set(np.nonzero(x_train[i])[0].tolist()) for i in range(len(x_train))]
        self.feature_size_per_class = []
        for i in range(len(feature_offsets)-1):
            self.feature_size_per_class.append(int(feature_offsets[i+1] - feature_offsets[i]))
        self.features_seen_per_class = [set() for _ in range(len(self.feature_size_per_class))]
        for i in idxs:
            f = self.x_features[i]
            k = self.y_train[i]
            self.features_seen_per_class[k] = self.features_seen_per_class[k].union(f)
        self.class_proportions = class_proportions
        self.feature_trials = feature_trials

    def sample(self, clf, log, t):
        scores = []
        for i in self.schedule:
            k = self.y_train[i]
            old_risk = self.class_proportions[k] * ((1 - len(self.features_seen_per_class[k]) / self.feature_size_per_class[k]) ** self.feature_trials)
            new_risk = self.class_proportions[k] * ((1 - len(self.features_seen_per_class[k].union(self.x_features[i])) / self.feature_size_per_class[k]) ** self.feature_trials)
            scores.append(new_risk - old_risk)
        top = self.schedule[np.argsort(scores)]
        idxs = top[:self.increment]
        self.schedule = np.setdiff1d(self.schedule, idxs)
        for i in idxs:
            f = self.x_features[i]
            k = self.y_train[i]
            self.features_seen_per_class[k] = self.features_seen_per_class[k].union(f)
        return idxs.tolist()

def get_phase1(idxs, x_train, y_train, options):
    if options.phase1 == 'fixed':
        return Passive(idxs, x_train, options.phase1_increment)
    elif options.phase1 == 'klogk':
        return Passive(idxs, x_train, options.phase1_increment)
    elif options.phase1 == 'until-all-labels':
        return Passive(idxs, x_train, options.phase1_increment)
    elif options.phase1 == 'm-per-class':
        return PerClass(x_train, y_train, options.m_per_class, options.phase1_increment)
    elif options.phase1 == 'skip':
        return
    else:
        raise Exception("%s not recognized " % options.phase1)

def get_phase2(idxs, x_train, y_train, options):
    if options.phase2 == 'passive':
        return Passive(idxs, x_train, options.phase2_increment)
    elif options.phase2 == 'margin':
        return Margin(idxs, x_train, options.phase2_increment)
    elif options.phase2 == 'entropy':
        return Entropy(idxs, x_train, options.phase2_increment)
    elif options.phase2 == 'greedy-coverage':
        return GreedyCoverage(idxs, x_train, y_train, options.phase2_increment, options.feature_offsets, options.class_props, options.l)
    elif options.phase2 == 'minipool-b':
        return MinipoolB(options.minipool_size, idxs, x_train, y_train, options.x_teach, options.y_teach, options.phase2_increment)
    elif options.phase2 == 'greedy':
        return GreedyAccuracy(idxs, x_train, y_train, options.x_teach, options.y_teach, options.phase2_increment)
    else:
        raise Exception("%s not recognized " % options.phase2)

def get_stopping_condition(y_train, options):
    if options.phase1 == 'fixed':
        return FixedStopping(options.fixed)
    elif options.phase1 == 'klogk':
        return KLogKStopping()
    elif options.phase1 == 'm-per-class':
        return FixedStopping(len(np.unique(y_train)) * options.m_per_class)
    elif options.phase1 == 'until-all-labels':
        return UntilAllLabelsStopping(y_train)
    elif options.phase1 == 'skip':
        return AlwaysAlreadyStopped()

def single_trial(x_train, y_train, x_test, y_test, options, trial):
    show_progress = True
    k = len(np.unique(y_train))
    big_n = options.N or np.shape(x_train)[0]
    idxs = []
    clf = LogisticRegression(C = options.regularizer, k=k)
    phase1 = get_phase1(idxs, x_train, y_train, options)
    stopping_condition = get_stopping_condition(y_train, options)
    stopped = False
    if show_progress:
        pbar = tqdm.tqdm(total=big_n)
    if stopping_condition.met(idxs):
        trial.info('stopped_phase1', len(idxs))
        phase2 = get_phase2(idxs, x_train, y_train, options)
        stopped = True
    lastn = 0
    while len(idxs) < big_n:
        if show_progress:
            pbar.update(len(idxs) - lastn)
        lastn = len(idxs)
        start_time = time.time()
        if trial.level_is_at_least(logging.DEBUG):
            oldclf = copy.deepcopy(clf)
        query_time = time.time()
        if not stopped:
            new_idxs = phase1.sample(clf, trial, len(idxs))
            idxs += new_idxs
            if stopping_condition.met(idxs):
                trial.info('stopped_phase1', len(idxs))
                phase2 = get_phase2(idxs, x_train, y_train, options)
                stopped = True
        else:
            new_idxs = phase2.sample(clf, trial, len(idxs))
            idxs += new_idxs
        n = len(idxs)
        trial.query_time.info(n, time.time() - query_time)
        clf = LogisticRegression(C = options.regularizer, k=k)
        training_time = time.time()
        clf.fit(x_train[idxs,:], y_train[idxs])
        trial.training_time.info(n, time.time() - training_time)
        test_time = time.time()
        predicted = clf.predict(x_test)
        err = 1 - np.mean(predicted == y_test)
        trial.test_time.info(n, time.time() - test_time)
        trial.z.info(n, new_idxs)
        trial.err.info(n, err)
        if trial.level_is_at_least(logging.DEBUG):
            trial.pretrain_model.debug(n, oldclf)
            trial.mean_predicted.debug(n, np.mean(predicted))
            trial.norm_beta.debug(n, np.linalg.norm(clf.coef_[0]))
            train_predicted = clf.predict(x_train[idxs,:])
            train_err = 1 - np.mean(train_predicted == y_train[idxs])
            trial.train_err.debug(n, train_err)
            trial.mean_train_predicted.debug(n, np.mean(train_predicted))
            trial.mean_train_ys.debug(n, np.mean(y_train[idxs]))
            trial.predicted.debug(n, predicted)
            trial.predicted_proba.debug(n, clf.predict_proba(x_test))
            trial.beta.debug(n, clf.coef_[0])
            trial.train_predicted.debug(n, train_predicted)
            trial.train_predicted_proba.debug(n, clf.predict_proba(x_train[idxs,:]))
    elapsed_time = time.time() - start_time
    trial.info('elapsed_time', elapsed_time)
    if show_progress:
        pbar.close()
    return trial

import numpy as np
import pandas as pd
import argparse
import bal
import log

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(title='Methods',dest='method')
parser.add_argument('TRAINING_SET', help="Training set file")
parser.add_argument('TEST_SET', help="Test set file")
parser.add_argument('OUTPUT_FILE', help="Output file")
parser.add_argument('--trials', type=int, default=20, help="Number of experimental trials")
parser.add_argument('--regularizer', type=float, default=1.0)
parser.add_argument('--batch', type=int, default=1, help="Batch size of training set")
parser.add_argument('--training-set-max-size', type=int, default=None, help="Maximum training set size")
passive_parser = subparsers.add_parser('passive')
active_parser = subparsers.add_parser('active')
active_parser.add_argument('--uncertainty', choices=['margin','entropy'],default='margin')
weak_teaching_parser = subparsers.add_parser('weak-teaching')
weak_teaching_parser.add_argument('--uncertainty', choices=['margin','entropy'],default='margin')
minipool_parser = subparsers.add_parser('minipool')
minipool_parser.add_argument('--validation-set', help="Validation set file")
minipool_parser.add_argument('--candidates', type=int, required=True, help="Number of candidates to use per iteration when using minipool")

def main(args):
    args.training_set = args.TRAINING_SET
    args.test_set = args.TEST_SET
    args.output = args.OUTPUT_FILE
    x_train, y_train = np.load(args.training_set)
    x_test, y_test = np.load(args.test_set)
    if args.training_set_max_size is None:
        args.training_set_max_size = np.shape(x_train)[0]
    args.N = args.training_set_max_size
    if args.method == 'passive':
        args.phase1 = 'fixed'
        args.fixed = 0
        args.phase1_increment = 0
        args.phase2 = 'passive'
    elif args.method == 'active':
        args.phase1 = 'until-all-labels'
        args.phase1_increment = args.batch
        args.phase2 = args.uncertainty
    elif args.method == 'weak-teaching':
        args.phase1 = 'm-per-class'
        args.phase1_increment = args.batch
        args.phase2 = args.uncertainty
        args.m_per_class = 1
    elif args.method == 'minipool':
        args.phase1 = 'fixed'
        args.fixed = 0
        args.phase1_increment = 0
        args.phase2 = 'minipool-b'
        args.minipool_size = args.candidates
        x_teach, y_teach = np.load(args.validation_set)
        args.x_teach = x_teach
        args.y_teach = y_teach
    args.phase2_increment = args.batch
    logger = log.Log()
    for i in range(args.trials):
        single_trial(x_train, y_train, x_test, y_test, args, logger.trials.select(i+1))
    df1 = logger.trials._.err.df()
    df2 = logger.trials._.stopped_phase1.df().transpose()
    df3 = logger.trials._.z.df()
    df3 = df3.apply(lambda x: pd.Series([z for y in x for z in y]), axis=0)
    df3.index += 1
    df_list = [df1,df2,df3]
    writer = pd.ExcelWriter(args.output, engine='xlsxwriter')
    df1.to_excel(writer,'Test Set Error')
    df2.to_excel(writer,'Initialization Complete At')
    df3.to_excel(writer,'Training Set Items')
    writer.save()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
