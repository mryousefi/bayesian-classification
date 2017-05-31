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

function Classifiers = risk_resub(Params, Model, Classifiers, lambda, x, n)

if Params.OBC.do
    count_obc = find_confusion_matrix(Params, Model, Classifiers.OBC, lambda, x);
    Classifiers.OBC.resub = find_risk(Model, lambda, count_obc, n);
end

if Params.LDA.do
    if ~Classifiers.LDA.error
        count_lda = find_confusion_matrix(Params, Model, Classifiers.LDA, lambda, x);
        Classifiers.LDA.resub = find_risk(Model, lambda, count_lda, n);
    else
        Classifiers.LDA.resub = NaN;
    end
end

if Params.QDA.do
    if ~Classifiers.QDA.error
        count_qda = find_confusion_matrix(Params, Model, Classifiers.QDA, lambda, x);
        Classifiers.QDA.resub = find_risk(Model, lambda, count_qda, n);
    else
        Classifiers.QDA.resub = NaN;
    end
end

if Params.LSVM.do
    if ~Classifiers.LSVM.error
        count_lsvm = find_confusion_matrix(Params, Model, Classifiers.LSVM, lambda, x);
        Classifiers.LSVM.resub = find_risk(Model, lambda, count_lsvm, n);
    else
        Classifiers.LSVM.resub = NaN;
    end
end

if Params.RBFSVM.do
    if ~Classifiers.RBFSVM.error
        count_rbfsvm = find_confusion_matrix(Params, Model, Classifiers.RBFSVM, lambda, x);
        Classifiers.RBFSVM.resub = find_risk(Model, lambda, count_rbfsvm, n);
    else
        Classifiers.RBFSVM.resub = NaN;
    end
end