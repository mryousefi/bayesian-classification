# bayesian-classification
## Description
This is a Bayesian Multi-class Classification and Risk Estimation Toolbox written in Matlab for the paper "[On optimal Bayesian classification and risk estimation under multiple classes](https://link.springer.com/article/10.1186/s13637-015-0028-3)" by L. A. Dalton and M. R. Yousefi published in the EURASIP Journal on Bioinformatics and Systems Biology in December 2015.

The optimal Bayesian classification paradigm addresses sample discrimination problems in a Bayesian framework by assuming that the true distributions characterizing multiple classes in the population are members of an uncertainty class of models weighted by a prior distribution. After observing the sample, the prior is updated to a posterior distribution, and the posterior predictive distributions (effective densities) on the sample space are found. 

This toolbox implements tools for several Gaussian models with known, independent scaled identity, independent arbitrary, homoscedastic scaled identity, and homoscedastic arbitrary covariances.  All models also assume unknown means and use conjugate priors. This toolbox includes implementations of the following for these models:

1. **Optimal Bayesian risk classifiers (OBRC)**, which reduce to the optimal Bayesian classifier (OBC) under binary classification with zero-one risk functions,

2. **Optimal Bayesian risk estimators (BRE)** under the OBRC and several other (linear and non-linear) classifiers, which reduce to optimal Bayesian error estimators under binary classification with zero-one risk functions, 

3. **Mean-square error (MSE)** of the BRE and several other risk estimators under the OBRC and several other (linear and non-linear) classifiers.

This toolbox uses a fast analytical form for the BRE and conditional MSE under binary linear classification. Of particular interest are the analytic forms for the MSE under Gaussian models with homoscedastic covariances.  Also of interest are the computationally efficient methods implemented to approximate the BRE and MSE when analytic expressions are not available. Several scripts demonstrating implementation of the above tools are provided.  

Below are some examples of several classifiers and their performance.

![Example model 1](https://github.com/mryousefi/bayesian-classification/blob/master/figs/model_1.gif)

![Example model 2](https://github.com/mryousefi/bayesian-classification/blob/master/figs/model_2.gif)

![Example decision boundaries for model 1](https://github.com/mryousefi/bayesian-classification/blob/master/figs/fig_1.gif)

Example decision boundaries for model 1: **(a)** LDA; **(b)** QDA; **(c)** OBRC; **(d)** L-SVM; **(e)** RBF-SVM. 

![Example decision boundaries for model 2](https://github.com/mryousefi/bayesian-classification/blob/master/figs/fig_2.gif)

Example decision boundaries for model 2: **(a)** LDA; **(b)** QDA; **(c)** OBRC; **(d)** L-SVM; **(e)** RBF-SVM. 

![True risk statistics](https://github.com/mryousefi/bayesian-classification/blob/master/figs/fig_3.gif)

True risk statistics for models 1 and 2 and five classification rules (LDA, QDA, OBRC, L-SVM, and RBF-SVM). **(a)** Model 1, mean; **(b)** model 1, standard deviation; **(c)** model 2, mean; **(d)** model 2, standard deviation.

## Dependecies
This toolbox requires the LIBSVM Matlab library, which should be copied in the 'libsvm' folder in the current project directory.

You can find the LIBSVM library [here](https://www.csie.ntu.edu.tw/~cjlin/libsvm/).

## How to start
You can start by playing with the example files, and then run them in Matlab:
- [main_arbit_homo_multi.m](https://github.com/mryousefi/bayesian-classificationb/blob/master/main_arbit_homo_multi.m)
- [main_arbit_homo_risk.m](https://github.com/mryousefi/bayesian-classification/blob/master/main_arbit_homo_risk.m)
- [main_arbit_indep_multi.m](https://github.com/mryousefi/bayesian-classification/blob/master/main_arbit_indep_multi.m)
- [main_arbit_indep_risk.m](https://github.com/mryousefi/bayesian-classification/blob/master/main_arbit_indep_risk.m)
- [main_iden_indep_multi_higherr.m](https://github.com/mryousefi/bayesian-classification/blob/master/main_iden_indep_multi_higherr.m)
- [main_iden_indep_multi_lowerr.m](https://github.com/mryousefi/bayesian-classification/blob/master/main_iden_indep_multi_lowerr.m)
- [main_iden_indep_risk_higherr.m](https://github.com/mryousefi/bayesian-classification/blob/master/main_iden_indep_risk_higherr.m)
- [main_iden_indep_risk_lowerr.m](https://github.com/mryousefi/bayesian-classification/blob/master/main_iden_indep_risk_lowerr.m)
