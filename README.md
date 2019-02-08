#### Active vs Passive Learning 
‘active_vs_passive.py’ performs text classification on **20 Newsgroups** Dataset using **Logistic Regression** and plots error curves for Active Learning and Passive Learning.  

‘regulariser’ parameter for Logistic Regression can be passed as an argument. (default = 0.1)

Classifier Training is divided into two phases. Until the stopping condition is met, training continues in phase 1.

Phase 1 options (default = ‘fixed’) :-
* fixed’: Stopping condition is met when the training set size is equal to a fixed size. The fixed size(default=1) can be passed as an argument to the code.
* ‘klogk’ : Stopping condition is met when the training set size is equal to klogk, where k corresponds to the total number of classes in dataset.
* 'until-all-labels' : Stopping condition is met when the training set has at least one training item from each class.
* ‘m-per-class’ :  The training set has at least ‘m’ training items from each class, before proceeding to phase 2. ‘m-per-class’ **should be** passed as an argument.

Phase 2 sampling methods:-
* ‘margin’ : Smallest Margin based Active Learning
* ‘entropy’ : Maximum Entropy based Active Learning 
* ‘passive’ : Passive Learning

‘phase1-increment’ and ‘phase2-increment’ : The growth rate of training set size for phase 1 and phase 2 can be passed as an argument. (default=1 for both)

The model is trained for varying training set size and the error on the test set is calculated for each of them, and this is repeated for multiple trials(Number of trials can be passed as argument. default=20). The code plots minimum, maximum and median of test error on each training set size over multiple trials. 

To run the code with the default configuration, run ‘python active_vs_passive.py’.

