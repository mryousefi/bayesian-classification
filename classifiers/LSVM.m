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

function lsvm = LSVM(Params, training, group)

N = length(group);
lsvm.SVMmean = mean(training, 1);
lsvm.SVMmeanstdv = sqrt(mean(var(training, [], 1)));

% data needs to be normalized to have a range approximately -1 to 1
training = (training - repmat(lsvm.SVMmean, N, 1))/lsvm.SVMmeanstdv;

try
    SVM = svmtrain(group, training, '-t 0 -q');
    lsvm.error = false;
    lsvm.model = SVM;
catch
    lsvm.error = true;
    lsvm.model = [];
end

lsvm.type = 'lsvm';

if ~lsvm.error
    n_start = 1;
    a = cell(Params.M);
    for i = 1:Params.M
        j_temp = 1;
        for j = 1:Params.M
            if i == j
                a{i, j} = [];
            else
                a{i, j} = SVM.SVs(n_start:(n_start + SVM.nSV(i) - 1), :)' * SVM.sv_coef(n_start:(n_start + SVM.nSV(i) - 1), j_temp)/lsvm.SVMmeanstdv;
                j_temp = j_temp + 1;
            end
        end
        n_start = n_start + SVM.nSV(i);
    end
    
    lsvm.a = cell(Params.M);
    lsvm.b = cell(Params.M);
    cnt = 1;
    for i = 1:Params.M-1
        for j = i+1:Params.M
            lsvm.a{i, j} = a{i, j} + a{j, i};
            lsvm.a{j, i} = -lsvm.a{i, j};
            lsvm.b{i, j} = -SVM.rho(cnt) - lsvm.SVMmean*lsvm.a{i, j};
            lsvm.b{j, i} = - lsvm.b{i, j};
            cnt = cnt + 1;
        end
    end
end