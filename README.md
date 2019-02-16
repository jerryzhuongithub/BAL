# Beyond Active Learning 
## Introduction

*Beyond Active Learning* is a Python library which implements various active learning algorithms. Sample experiments perform text classification on the **20 Newsgroups** dataset using a logistic regression learner over a binary bag of words feature representation, plotting error curves over varying training set sizes, and benchmarking them against passive learning.

The core functions - model training, testing and error plots - are implemented in [`main.py`](main.py).

To see an example of programmatic usage and to recreate the results in the
[`figs`](figs) folder, run the experimental script is
[`run_experiments.py`](run_experiments.py).

Options are set on a global `options` and may be set either programmatically or by command-line flags.

*Beyond Active Learning* implements the following active learning algorithms:

*   Margin based uncertainty sampling
*   Maximum entropy based uncertainty sampling

The programmatic entry point is that `training` method defined in [`main.py`](main.py). The method performs implements an active learner by sampling without replacement over a provided training set `x_train`, a NumPy input matrix, and `y_train`, a NumPy vector of labels. Evaluation is performed on the data set specified by the arguments `x_test, ` and `y_test`.

The active learner is specified by selecting its protocol during an initialization step and then its definition of uncertainty used during its active learning step. In the options, these are known, respectively, as `phase1` and `phase2`.

This learning procedure is repeated for the number of trials given in `options.trials` (or the `--trials` command-line flag). The `training` method returns a tuple of `(test_errors, stopped_at)`, whose first item is a matrix of test set errors at each iteration over the trials and second item is a list of the iteration at which the initialization step ends.

## Running experiments

The following options are defined in [`main.py`](main.py)


 Options |  Usage and Default Value        
---------| -------------------------
--regulariser  | Regularisation parameter for Logistic Regression. (default = 0.1)
--phase1 | 'Fixed', 'klogk', 'until-all-labels' or 'm-per-class' can be selected for phase 1. (default = 'until-all-labels')
--m-per-class | Number of training items 'm' required from each class. (default=1)
--fixed | Fixed size of training items if 'fixed' selected for phase1. (default = 1)
--phase2 | ‘Margin’, 'entropy' or 'passive' can be selected for phase 2. (default = 'passive')
--phase1-increment | Growth rate of training set size for phase 1. (default = 1)
--phase2-increment | Growth rate of training set size for phase 2. (default = 1)
--trials | Number of trials of training. (default = 20)

Available choices for argument `--phase1`
* 'fixed' : Initialize by sampling without replacement up to a fixed size (set by `fixed` parameter)
* 'klogk' :" Initialize by sampling without replacement k log k items where k corresponds to the number of classes in dataset.
* 'until-all-labels' : Initialize by sampling without replacement until the training set contains at least one training item from each class.
* 'm-per-class' : Initialize by sampling without replacement `m-per-class` training items from each class. 

Available choices for argument `--phase2`
* ‘margin’ : Smallest margin based uncertainty sampling
* ‘entropy’ : Maximum entropy based uncertainty sampling
* ‘passive’ : Passive learning (uniform sampling)

The options can be modified in [`run_experiments.py`](run_experiments.py) as needed.

To run the experiment, run:

```
$ python run_experiments.py
```

**Note : 'm-per-class' is not yet implemented**


## Results
Error plots can be found in 'figs/' folder. To reproduce the results, run :
```
$ ./run_experiments.sh
```





