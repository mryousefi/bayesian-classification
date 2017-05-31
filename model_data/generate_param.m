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

function [c, mu, Sigma] = generate_param(Params, Model)

switch Model.c.type
    case 'known'
        c = Model.c.value;
    case 'dirichlet'
        c = gamrnd(Model.c.alpha, 1);
        c = c/sum(c);
    otherwise
        disp('Error: invalid c model type (function generate_param).')
end

mu = zeros(Params.M, Params.d);
Sigma = cell(Params.M, 1);
switch Model.dist.structure
    case 'fixed'
        for i = 1:Params.M
            mu(i, :) = Model.dist.mu(i, :);
            Sigma{i} = Model.dist.Sigma{i};
        end
    case 'known'
        for i = 1:Params.M
            Sigma{i} = Model.dist.Sigma{i};
        end
        for i = 1:Params.M
            mu(i, :) = mvnrnd(Model.dist.m(i, :), Sigma{i}/Model.dist.nu(i));
        end
    case 'identity'
        switch Model.dist.type
            case 'indep'
                for i = 1:Params.M
                    sigma2 = iwishrnd(trace(Model.dist.S{i}), (Model.dist.kappa(i) + Params.d +1)*Params.d - 2);
                    Sigma{i} = sigma2*eye(Params.d);
                end
            case 'homo'
                sigma2 = iwishrnd(trace(Model.dist.S), (Model.dist.kappa + Params.d +1)*Params.d - 2);
                Sigma{1} = sigma2*eye(Params.d);
                for i = 2:Params.M
                    Sigma{i} = Sigma{1};
                end
            otherwise
                disp('Error: invalid covariance model type (function generate_param, identity case).')
        end
        mu = zeros(Params.M, Params.d);
        for i = 1:Params.M
            mu(i, :) = mvnrnd(Model.dist.m(i, :), Sigma{i}/Model.dist.nu(i));
        end
    case 'arbitrary'
        switch Model.dist.type
            case 'indep'
                for i = 1:Params.M
                    Sigma{i} = iwishrnd(Model.dist.S{i}, Model.dist.kappa(i));
                end
            case 'homo'
                Sigma{1} = iwishrnd(Model.dist.S, Model.dist.kappa);
                for i = 2:Params.M
                    Sigma{i} = Sigma{1}';
                end
            otherwise
                disp('Error: invalid covariance model type (function generate_param, arbitrary case).')
        end
        mu = zeros(Params.M, Params.d);
        for i = 1:Params.M
            mu(i, :) = mvnrnd(Model.dist.m(i, :), Sigma{i}/Model.dist.nu(i));
        end
    otherwise
        disp('Error: invalid covariance model structure (function generate_param).')
end