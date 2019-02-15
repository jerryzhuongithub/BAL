from main import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def train_20newsgroups_all():
    options.regularizer = 1.0
    corpus = fetch_20newsgroups(subset='all')
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(corpus.data)
    y = corpus.target
    indices = np.arange(len(y))
    x_train, x_rest, y_train, y_rest, idx_train, idx_rest = train_test_split(X, y, indices, test_size=0.67)
    x_teach, x_test, y_teach, y_test, idx_teach, idx_test = train_test_split(x_rest, y_rest, idx_rest, test_size=0.5)
    options.N = option('N', type=int, default=1000)
    options.phase1_increment = 10
    options.phase2_increment = 10
    options.phase2 = 'margin'
    test_err, stopped_phase1_at = training(20, x_train, y_train, x_test, y_test)
    plot_minmedianmax(test_err, stopped_phase1_at, folder='figs/active', title='20newsgroups-margin', filename='20newsgroups-margin.png', save=True)
    options.phase2 = 'entropy'
    test_err, stopped_phase1_at = training(20, x_train, y_train, x_test, y_test)
    plot_minmedianmax(test_err, stopped_phase1_at, folder='figs/active', title='20newsgroups-entropy', filename='20newsgroups-entropy.png', save=True)
    options.phase2 = 'passive'
    test_err, stopped_phase1_at = training(20, x_train, y_train, x_test, y_test)
    plot_minmedianmax(test_err, stopped_phase1_at, folder='figs/passive', title='20newsgroups-passive', filename='20newsgroups-passive.png', save=True)


if __name__ == '__main__':
    train_20newsgroups_all()