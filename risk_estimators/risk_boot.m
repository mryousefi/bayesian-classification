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

function Classifiers = risk_boot(Params, Model, Classifiers, lambda, x, n, R)

n_start = 1;
y = NaN(Params.train.n, 2);
for i = 1:Params.M
    y(n_start:(n_start + n(i) - 1), :) = [(1:n(i))', i*ones(n(i), 1)];
    n_start = n_start + n(i);
end

N = (1:Params.train.n)';
n_ts_tot = zeros(Params.M, 1);

if Params.OBC.do
    obc_count = zeros(Params.M);
end

if Params.LDA.do
    lda_count = zeros(Params.M);
end

if Params.QDA.do
    qda_count = zeros(Params.M);
end

if Params.LSVM.do
    lsvm_count = zeros(Params.M);
end

if Params.RBFSVM.do
    rbfsvm_count = zeros(Params.M);
end

lda_error = false;
qda_error = false;
lsvm_error = false;
rbfsvm_error = false;

[~, bootsam] = bootstrp(R, [], y);
for i = 1:R
    Prior = Model;
    trIdx = bootsam(:, i);
    tsIdx = setdiff(N, unique(trIdx));
    
    n_tr = zeros(Params.M, 1);
    n_ts = zeros(Params.M, 1);
    x_tr = cell(Params.M, 1);
    x_ts = cell(Params.M, 1);
    for j = 1:Params.M
        x_tr{j} = x{j}(y(trIdx(y(trIdx, 2) == j), 1), :);
        x_ts{j} = x{j}(y(tsIdx(y(tsIdx, 2) == j), 1), :);
        n_tr(j) = size(x_tr{j}, 1);
        n_ts(j) = size(x_ts{j}, 1);
    end
    
    % Update prior to posterior
    Posterior = update_prior(Params, Prior, x_tr, n_tr);
    
    % Train classifiers
    temp_Classifiers = classifier_design(Params, Posterior, lambda, x_tr, n_tr);
    
    % Count errors
    if Params.OBC.do
        obc_count = obc_count + find_confusion_matrix(Params, Posterior, temp_Classifiers.OBC, lambda, x_ts);
    end
    
    if Params.LDA.do
        if ~temp_Classifiers.LDA.error
            lda_count = lda_count + find_confusion_matrix(Params, Posterior, temp_Classifiers.LDA, lambda, x_ts);
        else
            lda_error = true | lda_error;
        end
    end
    
    if Params.QDA.do
        if ~temp_Classifiers.QDA.error
            qda_count = qda_count + find_confusion_matrix(Params, Posterior, temp_Classifiers.QDA, lambda, x_ts);
        else
            qda_error = true | qda_error;
        end
    end
    
    if Params.LSVM.do
        if ~temp_Classifiers.LSVM.error
            lsvm_count = lsvm_count + find_confusion_matrix(Params, Posterior, temp_Classifiers.LSVM, lambda, x_ts);
        else
            lsvm_error = true | lsvm_error;
        end
    end
    
    if Params.RBFSVM.do
        if ~temp_Classifiers.RBFSVM.error
            rbfsvm_count = rbfsvm_count + find_confusion_matrix(Params, Posterior, temp_Classifiers.RBFSVM, lambda, x_ts);
        else
            rbfsvm_error = true | rbfsvm_error;
        end
    end
    
    n_ts_tot = n_ts_tot + n_ts;
end

% calculate risk
if Params.OBC.do
    Classifiers.OBC.boot = find_risk(Model, lambda, obc_count, n_ts_tot);
    Classifiers.OBC.boot632 = 0.632*Classifiers.OBC.boot + 0.368*Classifiers.OBC.resub;
end

if Params.LDA.do
    if ~lda_error
        Classifiers.LDA.boot = find_risk(Model, lambda, lda_count, n_ts_tot);
        Classifiers.LDA.boot632 = 0.632*Classifiers.LDA.boot + 0.368*Classifiers.LDA.resub;
    else
        Classifiers.LDA.boot = NaN;
        Classifiers.LDA.boot632 = NaN;
    end
end

if Params.QDA.do
    if ~qda_error
        Classifiers.QDA.boot = find_risk(Model, lambda, qda_count, n_ts_tot);
        Classifiers.QDA.boot632 = 0.632*Classifiers.QDA.boot + 0.368*Classifiers.QDA.resub;
    else
        Classifiers.QDA.boot = NaN;
        Classifiers.QDA.boot632 = NaN;
    end
end

if Params.LSVM.do
    if ~lsvm_error
        Classifiers.LSVM.boot = find_risk(Model, lambda, lsvm_count, n_ts_tot);
        Classifiers.LSVM.boot632 = 0.632*Classifiers.LSVM.boot + 0.368*Classifiers.LSVM.resub;
    else
        Classifiers.LSVM.boot = NaN;
        Classifiers.LSVM.boot632 = NaN;
    end
end

if Params.RBFSVM.do
    if ~rbfsvm_error
        Classifiers.RBFSVM.boot = find_risk(Model, lambda, rbfsvm_count, n_ts_tot);
        Classifiers.RBFSVM.boot632 = 0.632*Classifiers.RBFSVM.boot + 0.368*Classifiers.RBFSVM.resub;
    else
        Classifiers.RBFSVM.boot = NaN;
        Classifiers.RBFSVM.boot632 = NaN;
    end
end