import numpy as np
from sklearn.model_selection import GridSearchCV
import scipy as sp
import scipy.optimize
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator


class LogisticRegression(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    def __init__(self, C=1e4, tol=1e-4, k=None):
        self.C = C
        self.tol = tol
        self.k = k
        self.coef_ = None
        self.intercept_ = None
        self.funcalls_lastfit = None

    def fit(self, X, y):
        if self.k:
            self.classes_ = np.array(range(self.k))
            k = self.k
        else:
            self.classes_ = np.sort(np.unique(y))
            k = len(self.classes_)
        n = len(y)
        d = X.shape[1]
        intercept = np.ones(n).reshape(n, 1)
        if isinstance(X, np.ndarray):
            X_ = np.concatenate((intercept, X), axis = 1)
        else:
            X_ = sp.sparse.hstack((intercept, X))
        self.coef_ = self.coef_ if self.coef_ is not None else np.zeros((k, d))
        self.intercept_ = self.intercept_ if self.intercept_ is not None else np.zeros(k)
        for idx, i in enumerate(self.classes_):
            x0 = np.zeros(d + 1)
            x0[0] = self.intercept_[idx]
            x0[1:] = self.coef_[idx]
            yi = 1 * (y == i)
            optimLogitLBFGS = sp.optimize.fmin_l_bfgs_b(self.f,
                                                        x0 = x0,
                                                        args = (X_, yi),
                                                        fprime = self.fprime,
                                                        pgtol = self.tol)
            self.funcalls_lastfit = optimLogitLBFGS[2]['funcalls']
            beta = optimLogitLBFGS[0]
            self.intercept_[idx] = beta[0]
            self.coef_[idx] = beta[1:]

    def f(self, w, X, y):
        s = np.log((1.0 + np.exp(safe_sparse_dot(X, w))))
        ll = 0.5 * (1.0/self.C) * safe_sparse_dot(w,w) - (np.sum(y*(safe_sparse_dot(X, w) - s) + (1-y)*(-s)))
        return ll

    def logit(self, X, w):
        return np.exp(safe_sparse_dot(X, w))/(1.0 + np.exp(safe_sparse_dot(X, w)))

    def fprime(self, w, X, y):
        return((1.0/self.C) * w + safe_sparse_dot(X.T, (self.logit(X, w) - y)))

    def predict_proba(self, X):
        prob = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob


class LogisticRegressionCV(LogisticRegression):
    def __init__(self, tol=1e-4, cv=None):
        self.tol = tol
        self.cv = cv

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        k = len(self.classes_)
        d = X.shape[1]
        self.coef_ = np.zeros((k, d))
        self.intercept_ = np.zeros(k)
        self.C_ = np.zeros(k)
        Cs = np.power(10,np.linspace(-4,4,10))
        parameters = {'C': Cs}
        for idx, i in enumerate(self.classes_):
            gs = GridSearchCV(LogisticRegression(tol=self.tol), parameters, cv=5)
            yi = 1 * (y == i)
            gs.fit(X,yi)
            scores = gs.cv_results_['mean_test_score']
            best_index = scores.sum(axis=0).argmax()
            self.C_[idx] = Cs[best_index]
            clf = LogisticRegression(C=self.C_[idx], tol=self.tol)
            clf.fit(X,yi)
            self.intercept_[idx] = clf.intercept_[0]
            self.coef_[idx] = clf.coef_[0]
