# Beyond Active Learning 
## Introduction
This python module compares performance of passive learning and active learning. It performs text classification on **20 Newsgroups** dataset using Logistic Regression and plots error curves for varying training set sizes.

Few key components for running the experiments :
*   Main experimental script is in [`run_experiments.py`](run_experiments.py) with many options.
*   Core implementation - model training, testing and error plots - is implemented in [`main.py`](main.py).

Available Sampling methods for Active Learning:
*   Margin based uncertainty sampling
*   Maximum entropy based uncertainty sampling

Training phase is divided into two parts. Until the stopping condition is met, training continues in phase 1. The model is trained for varying training set sizes and the error on the test set is calculated for each of them, and this is repeated for multiple trials. The code plots minimum, maximum and median of test error on each training set size over multiple trials. 


## Running experiments

The following are the options that can be set in [`run_experiments.py`](run_experiments.py)

 Options |  Usage and Default Value        
---------| -------------------------
--regulariser  | Regularisation parameter for Logistic Regression. (default = 0.1)
--phase1 | 'Fixed', 'klogk', 'until-all-labels' or 'm-per-class' can be selected for phase 1. (default = 'until-all-labels')
--m-per-class | Number of training items 'm' required from each class. This **should be** passed as an argument if 'm-per-class' selected for phase1
--fixed | Fixed size of training items if 'fixed' selected for phase1. (default = 1)
--phase2 | ‘Margin’, 'entropy' or 'passive' can be selected for phase 2. (default = 'passive')
--phase1-increment | Growth rate of training set size for phase 1. (default = 10)
--phase2-increment | Growth rate of training set size for phase 2. (default = 10)
--trials | Number of trials of training. (default = 20)

Available choices for argument '--phase1'/'--phase2'
* 'fixed' : Stopping condition is met when the training set size is equal to a fixed size
* 'klogk' :" Stopping condition is met when the training set size is equal to klogk, where k corresponds to the total number of classes in dataset.
* 'until-all-labels' : Stopping condition is met when the training set has at least one training item from each class.
* 'm-per-class' : The training set has at least ‘m’ training items from each class, before proceeding to phase 2. 
* ‘margin’ : Smallest margin based uncertainty sampling
* ‘entropy’ : Maximum entropy based uncertainty sampling
* ‘passive’ : Passive learning (uniform sampling)

To run the experiment with default configuration, run :

```
$ python run_experiments.py
```

## Results
Error plots can be found in 'figs/' folder. To reproduce the results, run :
```
$ ./run_experiments.sh
```





