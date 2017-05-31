% This file is part of a Bayesian Multi-class Classification and Risk 
% Estimation Toolbox
%
% On optimal Bayesian classification and risk estimation under multiple 
% classes
% By Lori A. Dalton and Mohammadmahdi R. Yousefi
% Published in EURASIP Journal on Bioinformatics and Systems Biology
% December 2015, 2015:8
%
% Copyright 2017,  L. A. Dalton and M. R. Yousefi
%
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, 
% this list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation 
% and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
% THE POSSIBILITY OF SUCH DAMAGE.
%========================================================================

% True Model: aribtrary and homoscedastic covariance matrices
% Assumed Model: aribtrary and homoscedastic covariance matrices
% Risk: NOT zero-one
% Classes: binary

% parpool(4)

% This toolbox requires LIBSVM matlab library to be located in a 'libsvm' 
% folder in the current directory.
% You can find the LIBSVM library here 
% https://www.csie.ntu.edu.tw/~cjlin/libsvm/
addpath('libsvm');
addpath('classifiers', 'model_data', 'risk_estimators', 'utils', 'validity_checks');

rng('shuffle');
% c.type: 'known', 'dirichlet'
% c.info: need > 0, 'low' = eps, 'high' = inf
% dist.structure: 'fixed', 'known', 'identity', 'arbitrary'
% dist.type: 'indep', 'homo'
% mean.info: need > 0, 'low' = eps, 'high' = inf
% cov.info: need > d - 1, 'low' = d - 1 + eps, 'high' = inf
% train.method: 'random', 'stratified'

M = 2;
d = 2;
max_rep = 2e3;
internal_rep = 1;
sample_size = [20, 60, 100, 140];
test_size = 10000;

true_mean_info = zeros(M, 1);
true_mean_info(1) = 12;
true_mean_info(2) = 2;
true_mean = zeros(M, d);
true_mean(1, :) = zeros(1, d);
true_mean(2, :) = 0.5*ones(1, d);
true_cov_info = zeros(M, 1);
true_cov_info(1) = 6;
true_cov_info(2) = 6;
true_cov = cell(M, 1);

assm_mean_info = zeros(M, 1);
assm_mean_info(1) = 12;
assm_mean_info(2) = 2;
assm_mean = zeros(M, d);
assm_mean(1, :) = zeros(1, d);
assm_mean(2, :) = 0.5*ones(1, d);

assm_cov_info = zeros(M, 1);
assm_cov_info(1) = 6;
assm_cov_info(2) = 6;
assm_cov = cell(M, 1);

for i = 1:M
    true_cov{i} = 0.3*eye(d);
    assm_cov{i} = 0.3*eye(d);
end

risk_matrix = [0 2; 1 0];
% risk_matrix = [0 1 1; 1 0 1; 1 1 0];
N = 1000000;

disp('start')

tic
% Simulation parameters
% parfor rep = 1:max_rep
for rep = 1:max_rep
    disp(['true-arbit-homo --- assumed-arbit-homo --- rep:', num2str(rep)]);
    [Params, TruePrior, AssumedPrior, Sample, Classifiers, Posterior, Results(rep)] = run_BRE(...
        'sample_size', sample_size, ...
        'test_size', test_size, ...
        'irep', internal_rep, ...
        'M', M, 'd', d, 'sampling', 'stratified', ...
        'true_c', 'known', 'true_c-info', 2, ...
        'true_structure', 'arbitrary', 'true_type', 'homo', ...
        'true_mean-info', true_mean_info, 'true_mean', true_mean, ...
        'true_cov-info', true_cov_info, 'true_cov', true_cov, ...
        'assm_c', 'known', 'assm_c-info', 2, ...
        'assm_structure', 'arbitrary', 'assm_type', 'homo', ...
        'assm_mean-info', assm_mean_info, 'assm_mean', assm_mean, ...
        'assm_cov-info', assm_cov_info, 'assm_cov', assm_cov, ...
        'n-synt', N, ...
        'lambda', risk_matrix, ...
        'classifiers', {'LDA', 'QDA', 'OBC'}, ...
        'error_estimators', {'TRUE', 'RESUB', 'LOO', 'CV', 'BOOT', 'BRE_EXACT', 'BRE_APPROX', 'MSE_EXACT', 'MSE_APPROX'});
end
toc

% delete(gcp);

save('results-arbit-homo-risk.mat', 'Results', 'max_rep', 'sample_size');