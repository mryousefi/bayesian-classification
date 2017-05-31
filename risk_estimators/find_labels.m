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

function labels = find_labels(Params, Model, Classifier, lambda, x)

% x{y, z} contains samples from class y.  The z-dimension is for cases
% where there may be several batchs of samples from class y, for instance,
% when finding the conditional MSE in the homoscedastic models.

y_max = size(x, 1);
z_max = size(x, 2);

labels = cell(y_max, z_max);
switch Classifier.type
    case 'obc'
        for y = 1:y_max
            for z = 1:z_max
                labels{y, z} = OBC_test(Params, Model, Classifier, lambda, x{y, z});
            end
        end
    case {'lda', 'qda'}
        for y = 1:y_max
            for z = 1:z_max
                labels{y, z} = discr_test(Classifier.model, x{y, z});
            end
        end
    case {'lsvm', 'rbfsvm'}
        for y = 1:y_max
            for z = 1:z_max
                labels{y, z} = SVM_test(Classifier, x{y, z});
            end
        end
    otherwise
        disp('Error: invalid classifier type (function risk_apparent).')
end

