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

function [risk_est, varepsilon] = BRE_exact(Params, Model, Classifier, lambda, obc)

a = Classifier.a{1, 2};
b = Classifier.b{1, 2};

if (strcmp(Classifier.type, 'lda') || strcmp(Classifier.type, 'lsvm')) && (Params.M == 2)
    varepsilon = NaN(Params.M, Params.M);
    switch Model.dist.structure
        case 'known'
            for y = 1:Params.M
				i = 1;
                m = (-1)^(i)*(obc.mu_eff(y, :)*a + b);
                c = a'*obc.Sigma_eff{y}*a;
                varepsilon(i, y) = normcdf(-m/sqrt(c));
                varepsilon(3-i, y) = 1 - varepsilon(i, y);
            end
        case {'identity', 'arbitrary'}
            for y = 1:Params.M
				i = 1;
                m = (-1)^(i)*(obc.mu_eff(y, :)*a + b);
                c = a'*obc.Sigma_eff{y}*a;
                varepsilon(i, y) = 0.5*(1 - sign(m)*betainc(m^2/(m^2 + obc.df(y)*c), 1/2, obc.df(y)/2));
                varepsilon(3-i, y) = 1 - varepsilon(i, y);
            end
        otherwise
            disp('Error: invalid covariance model structure (function BRE_exact).')
    end
    
    risk_cond = sum(varepsilon.*lambda, 1);
    risk_est = risk_cond*obc.c_eff;
else
    disp('Error: invalid classifier type (function BRE_exact).')
end