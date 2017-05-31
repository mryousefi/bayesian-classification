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

function qda = QDA(Params, Model, lambda, training, group)

switch Model.c.type
    case 'known'
        try
%             [sum(group==1), sum(group==2)]
            qda.model = fitcdiscr(training, group, 'Cost', lambda', 'DiscrimType', 'Quadratic', 'Prior', Model.c.value);
            qda.error = false;
        catch
%             [sum(group==1), sum(group==2)]
            qda.model = [];
            qda.error = true;
        end
    otherwise
        try
%             [sum(group==1), sum(group==2)]
            qda.model = fitcdiscr(training, group, 'Cost', lambda', 'DiscrimType', 'Quadratic', 'Prior', 'empirical');
            qda.error = false;
        catch
%             [sum(group==1), sum(group==2)]
            qda.model = [];
            qda.error = true;
        end
end

qda.type = 'qda';

if ~qda.error
    for i = 1:Params.M
        for j = 1:Params.M
            qda.A{i, j} = qda.model.Coeffs(i, j).Quadratic;
            qda.b{i, j} = qda.model.Coeffs(i, j).Linear;
            qda.c{i, j} = qda.model.Coeffs(i, j).Const;
        end
    end
end