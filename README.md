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
--regulariser  | Regularisation parameter for Logistic Regression. (default = `0.1`)
--phase1 | `fixed`, `klogk`, `until-all-labels` or `m-per-class` can be selected for phase 1. (default = `until-all-labels`)
--m-per-class | Number of training items `m` required from each class. (default=`1`)
--fixed | Fixed size of training items if `fixed` selected for phase1. (default = `1`)
--phase2 | `margin`, `entropy` or `passive` can be selected for phase 2. (default = `passive`)
--phase1-increment | Growth rate of training set size for phase 1. (default = `1`)
--phase2-increment | Growth rate of training set size for phase 2. (default = `1`)
--trials | Number of trials of training. (default = `20`)

Available choices for argument `--phase1`
* `fixed` : Initialize by sampling without replacement up to a fixed size (set by `fixed` parameter)
* `klogk` :" Initialize by sampling without replacement k log k items where k corresponds to the number of classes in dataset.
* `until-all-labels` : Initialize by sampling without replacement until the training set contains at least one training item from each class.
* `m-per-class` : Initialize by sampling without replacement `m-per-class` training items from each class. 

Available choices for argument `--phase2`
* `margin` : Smallest margin based uncertainty sampling
* `entropy` : Maximum entropy based uncertainty sampling
* `passive` : Passive learning (uniform sampling)

The options can be modified in [`run_experiments.py`](run_experiments.py) as needed.

To run the experiment, run:

```
$ python run_experiments.py
```

### Option Settings for Common Usage Scenarios

### Passive Learning

Passive learning samples without replacement from training pool of items through the entire training procedure.

 Options | Value
---------| -------------------------
--phase1 | `fixed`
--phase2 | `passive`

### Active Learning

Active learning queries without replacement from the training pool of items by choosing the available items which maximize some measure of uncertainty over their labels. The `margin` method chooses the `phase2-increment` item(s) with the least difference between the most likely class and the second most likely class, as predicted by the current classifer. The `entropy` method chosses the `phase2-increment` item(s) which maximize the entropy over the entire distribution of label predictions.

In the case of binary classification, the `margin` method and `entropy` method are equivalent. For multi-class classification, the recommended method is `margin`.

Active learning performance is improved by initialization which samples from the training pool (just as passive learning) until each label in the training pool is included in the training set at least once. This initialization strategy is configured by setting the option `phase1` to `until-all-labels`.

 Options | Value
---------| -------------------------
--phase1 | `until-all-labels`
--phase2 | `margin`

### Weak Teaching

Weak teaching uses the query policy of active learning. However, it initialized by choosing `m-per-class` items of each class in the training pool. Since this can only be done with knowledge of the true labels, weak teaching is distinct from active learning.

 Options      | Value
--------------| -------------------------
--phase1      | `m-per-class`
--phase2      | `margin`

The `m-per-class` option (distinct from the value `m-per-class` for the `phase1` option) controls the number of items drawn from each class during the initialization phase. The default is `1`.

## Results
Error plots can be found in 'figs/' folder. To reproduce the results, run :
```
$ ./run_experiments.sh
```





