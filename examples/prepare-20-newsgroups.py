import os
import sys
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

"""
Example script to demonstrate how to save data for bal.py command-line use.
"""

if len(sys.argv) >= 4:
    folder = sys.argv[1]
    corpus = fetch_20newsgroups(subset='all', categories=[sys.argv[2],sys.argv[3]])
else:
    folder = '20newsgroups'
    corpus = fetch_20newsgroups(subset='all')

vectorizer = CountVectorizer(binary=True)
x = vectorizer.fit_transform(corpus.data)
y = corpus.target
x_training, x_rest, y_training, y_rest = train_test_split(x, y, test_size=0.67)
x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest, test_size=0.5)

os.makedirs(folder,exist_ok=True)
np.save(os.path.join(folder, 'training.npy'), (x_training, y_training))
np.save(os.path.join(folder, 'test.npy'), (x_test, y_test))
np.save(os.path.join(folder, 'validation.npy'), (x_validation, y_validation))
