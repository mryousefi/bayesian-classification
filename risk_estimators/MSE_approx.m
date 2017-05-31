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

function risk_mse = MSE_approx(Params, Model, Classifier, lambda, obc, x, w, flag_risk_external, varepsilon_x, varepsilon_w)

if flag_risk_external == false
    [risk_est_x, varepsilon_x] = BRE_approx(Params, Model, Classifier, lambda, obc, x);
    [risk_est_w, varepsilon_w] = BRE_approx(Params, Model, Classifier, lambda, obc, w);
else
    risk_cond_x = sum(varepsilon_x.*lambda, 1);
    risk_est_x = risk_cond_x*obc.c_eff;
    risk_cond_w = sum(varepsilon_w.*lambda, 1);
    risk_est_w = risk_cond_w*obc.c_eff;
end

labels_x = find_labels(Params, Model, Classifier, lambda, x);
labels_w = find_labels(Params, Model, Classifier, lambda, w);

switch Model.dist.type
    case 'indep'
        term = NaN(Params.M);
        for y = 1:Params.M
            for z = y
                varepsilon_2 = NaN(Params.M);
                for i = 1:Params.M
                    for j = 1:Params.M
                        varepsilon_2(i, j) = mean((labels_x{y} == i) & (labels_w{y} == j));
                    end
                end
                term(y, y) = (lambda(:, y)')*varepsilon_2*(lambda(:, y));
            end
            for z = [1:(y-1), (y+1):Params.M]
                varepsilon_2 = NaN(Params.M);
                for i = 1:Params.M
                    for j = 1:Params.M
                        varepsilon_2(i, j) = varepsilon_x(i, y)*varepsilon_w(j, z);
                    end
                end
                term(y, z) = (lambda(:, y)')*varepsilon_2*(lambda(:, z));
            end
        end
        mom2 = sum(sum(obc.c_eff_cross.*term));
        risk_mse = mom2 - risk_est_x*risk_est_w;
    case 'homo'
        term = NaN(Params.M);
        for y = 1:Params.M
            for z = 1:Params.M
                varepsilon_2 = NaN(Params.M);
                for i = 1:Params.M
                    for j = 1:Params.M
                        varepsilon_2(i, j) = mean((labels_x{y, z} == i) & (labels_w{z, y} == j));
                    end
                end
                term(y, z) = (lambda(:, y)')*varepsilon_2*(lambda(:, z));
            end
        end
        mom2 = sum(sum(obc.c_eff_cross.*term));
        risk_mse = mom2 - risk_est_x*risk_est_w;
    otherwise
        disp('Error: invalid covariance model type (function MSE_approx).')
end

end
