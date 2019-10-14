# ROB 311 - TP4 - 07/10/2019

Authors: Simon Queyrut, Zhi Zhou
 
In this TP, we implement the algorithm SVM to classify digital numbers. Some main parameters are list below:
1. C: the C parameter tells the SVM optimization how much we want to avoid misclassifying each training example. For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane. We choose default value 1 in our case.
2. gamma: the gamma parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as the inverse of the radius of influence of samples selected by the model as support vectors. We choose default value *scale* in our case, which means it uses `1 / (n_features * X.var())` as value of gamma.
3. kernel: kernel parameters selects the type of hyperplane used to separate the data. Using ‘linear’ will use a linear hyperplane (a line in the case of 2D data). ‘rbf’ and ‘poly’ uses a non linear hyper-plane, we use rbf kernel in our case since images classification is non linear.

Since training all the data takes a long time, so we used 5000 training samples and trained for 3 epoches, finally we got a accuracy of **95.1%** on test data, which is quite good considering we use only a small percentage of the training data. 

Besides, our code enables users to find better parameters automatically, use `python svm.py -s True` to try this feature.

Important note
----

We uploaded the large training data file to github, this may cause some issues when using command `git clone`. If you encounter any problems when cloning the repo, please run `$ brew install git-lfs` or refer to the installation [website](https://git-lfs.github.com/) then clone using the `$ git lfs clone https://github.com/zroykhi/ROB311.git` command.
