from main import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

options.phase1 = option('phase1', choices=['klogk','fixed','until-all-labels','m-per-class'], default='m-per-class')
options.fixed = option('fixed', type=int, default=20)
options.phase1_increment = option('phase1-increment', type=int, default=10)
options.phase2_increment = option('phase2-increment', type=int, default=10)
options.N = option('N', type=int, default=1000)
options.trials = option('trials', type=int, default=20)


def train_20newsgroups_all():
    options.regularizer = 1.0
    corpus = fetch_20newsgroups(subset='all')
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(corpus.data)
    y = corpus.target
    indices = np.arange(len(y))
    x_train, x_rest, y_train, y_rest, idx_train, idx_rest = train_test_split(X, y, indices, test_size=0.67)
    x_teach, x_test, y_teach, y_test, idx_teach, idx_test = train_test_split(x_rest, y_rest, idx_rest, test_size=0.5)

    options.phase2 = 'margin'
    test_err, stopped_phase1_at = training(x_train, y_train, x_test, y_test)
    plot_minmedianmax(test_err, stopped_phase1_at, legend_name='active', folder='figs/active', title='20newsgroups-margin', filename='20newsgroups-margin.png', save=True)
    options.phase2 = 'entropy'
    test_err, stopped_phase1_at = training(x_train, y_train, x_test, y_test)
    plot_minmedianmax(test_err, stopped_phase1_at, legend_name='active', folder='figs/active', title='20newsgroups-entropy', filename='20newsgroups-entropy.png', save=True)
    options.phase1 = 'fixed'
    options.phase2 = 'passive'
    test_err, stopped_phase1_at = training(x_train, y_train, x_test, y_test)
    plot_minmedianmax(test_err, stopped_phase1_at, legend_name='passive', folder='figs/passive', title='20newsgroups-passive', filename='20newsgroups-passive.png', save=True)


if __name__ == '__main__':
    train_20newsgroups_all()