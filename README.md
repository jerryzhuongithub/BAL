# Beyond Active Learning 

## Introduction

*Beyond Active Learning* is a Python library which implements various active learning and machine teaching algorithms. Sample experiments perform text classification on the **20 Newsgroups** dataset using a logistic regression learner over a binary bag of words feature representation, plotting error curves over varying training set sizes, and benchmarking them against passive learning.

*Beyond Active Learning* implements the following learning algorithms:

* Passive learning (`passive`)
* Active learning with both margin- and entropy- based uncertainty sampling (`active`)
* Weak teaching with both margin- and entropy- based uncertainty sampling (`weak-teaching`)
* Teaching with a greedy accuracy heuristic (`minipool`)

## Usage

### 1. Prepare the datasets.

`BAL` accepts a data set in NumPy's NPY file format stored as a tuple `np.save((X,y))` for features `X` and labels `y`.

The following example code shows how to save a data set in the required format:

```
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

corpus = fetch_20newsgroups(subset='all')
vectorizer = CountVectorizer(binary=True)
x = vectorizer.fit_transform(corpus.data)
y = corpus.target
x_training, x_rest, y_training, y_rest = train_test_split(x, y, test_size=0.67)
x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest, test_size=0.5)
np.save('training.npy', (x_training, y_training))
np.save('test.npy', (x_test, y_test))
np.save('validation.npy', (x_validation, y_validation))
```

All methods require a training set and a test set. Some methods, as described below, will require a validation set.

### 2. Run the learning method.

`BAL` provides a command-line interface through `bal.py` and `plot.py`. The usage pattern is:

```
$ python bal.py METHOD ... TRAINING_SET TEST_SET OUTPUT_FILE ...
```

Here `method` names the algorithm to be used (`passive,active,weak-teaching,minipool}`).
The first ellipsis `...` is where method-specific parameters must be specified.
Then follow three positional arguments: `TRAINING_SET`, the name of the training set file saved as an `.NPY` as described above, `TEST_SET` for the `.NPY` test set, and `OUTPUT_FILE` for the results file, which will be an `.XLSX` spreadsheet format.
The second ellipsis `...` is where parameters general to all methods are specified.

As an example:

```
$ python bal.py active --uncertainty entropy training.npy test.npy results.xlsx --trials 20 --training-set-max-size 100
```
will perform active learning on `training.npy` using uncertainty-based active learning with entropy-based querying.
It will evaluate results on `test.npy` and save results in `results.xlsx`.
This procedure will be repeated for `20` trials for training set size up to `100`.

For usage on parameters general to all methods, see:
```
$ python bal.py -h
```

For information on method-specific command-line parameters, see (e.g.):

```
$ python bal.py active -h
```

### 3. Plot results.

To plot results, `plot.py` accepts a list of results files and the name of an output file and plots min-median-max curves for all results files provided at the command-line. For example:

```
python plot.py active.xlsx passive.xlsx err.png
```

will read `active.xlsx` and `passive.xlsx` which would have been created by two command-line invocations of `bal.py` as described above and plot them in `err.png`.

## Examples

The `/examples` folder contains examples from 20 Newsgroups of all supported methods. To run them, run `make` in the `/examples` folder.

## Description of Methods

### Passive Learning (`passive`)

Passive learning samples without replacement from training pool through the entire training procedure.

There are no method-specific parameters.


### Active Learning (`active`)

Active learning queries without replacement from the training pool of items by choosing the available items which maximize some measure of uncertainty over their labels.

There is one method-specific parameter `--uncertainty` with two values: `margin`, `entropy`.
The `margin` chooses the item at step `t` which maximizes the margin (or difference) between the most likely predicted class and the second most likely predicted class, as predicted by the classifier trained on the first `t-1` items.
The `entropy` method chooses the item which maximizes the entropy over the entire set of predicted labels.
For binary classification tasks, these methods are equivalent.
For multi-class classification, the recommended method is `margin`.
<!-- TODO impact of batch -->
<!-- TODO math up definitions -->

Active learning performance is improved by initialization which samples from the training pool (just as passive learning) until each label in the training pool is included in the training set at least once. Initialization is performed by performing passive learning until a single item from each class is observed. For example, if the task is contains 20-classes, active learning will sample without replacement from the training pool until at least one item from each of the 20 classes is observed. Let us say this occurs at step 90. The uncertainty-based active learning query strategy with then be first invoked in choosing the 91st (and all subsequent) items.

### Weak Teaching (`weak-teaching`)

Weak teaching uses the query policy of active learning. It differs however in its initialization strategy. Weak teaching is initialized by choosing an item of each class in the training pool. Since this can only be done with knowledge of the true labels, weak teaching is distinct from active learning.

It shares the model-specific parameter of `--uncertainty` with active learning methods.

### Minipool (`minipool`)

Minipool is a teaching algorithm which performs the following, in pseudocode:

* $current \leftarrow \emptyset$.
* $remaining \leftarrow \texttt{training_set}$
* while $size(current) \le \texttt{training_set_max_size}$:
  * for $j$ in $1$ .. $\texttt{candidates}$
    * $z_j \sim remaining$
    * $score_j \leftarrow error_{\texttt{validation_set}}(current \cup \{z_j\})$
  * $best \leftarrow$ $z_j$ for `batch` items with lowest $score_j$
  * $current = current \cup best$
  * $remaining = remaining \setminus best$
* return $current$
  
That is, minipool chooses `candidates` items out of the training set pool at each iteration, evaluates the error on a validation set after on the item appended to the current training set, and chooses the best `batch` to then add to the current training set.

Minipool requires the following method-specific parameters:

  Parameter       |  Description
------------------| -------------------------
---validation-set | .NPY filename for validation set required for evaluating minipool candidates
---candidates     | Number of candidates for Minipool to consider at each iteration

## General Parameters

The following command-line parameters are common to all methods:

Parameter                | Default | Description
-------------------------| --------| -----------------
---trials                | 20      | Number of trials to run
---regularizer           | 1e4     | Value of logistic regression regularizer (equivalent to `C` argument in `scikit-learn` for linear learners)
---batch                 | 1       | Number of items to add to training set at each iteration
---training-set-max-size | None    | Training set size to reach before concluding trial. Default concludes when entire training set is used.
