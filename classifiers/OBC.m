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

function obc = OBC(Params, Model)

% you might be able to specify special cases where obc is linear or
% quadratic later
obc.type = 'obc';

switch Model.c.type
    case 'known'
        obc.c_eff = Model.c.value;
		%obc.c_eff_2 = Model.c.value.^2;
		obc.c_eff_cross = Model.c.value*Model.c.value';
		%obc.c_eff_var = 0;
		%obc.c_eff_cov = 0;
    case 'dirichlet'
	    alpha = sum(Model.c.alpha);
        obc.c_eff = Model.c.alpha/alpha;
		%obc.c_eff_2 = Model.c.alpha.*(Model.c.alpha + 1)/(alpha*(alpha + 1));
		obc.c_eff_cross = Model.c.alpha*Model.c.alpha'/(alpha*(alpha + 1));
		obc.c_eff_cross(1:(Params.M+1):end) = Model.c.alpha/(alpha*(alpha + 1));
		%obc.c_eff_var = Model.c.alpha.*(alpha - Model.c.alpha)/(alpha^2*(alpha + 1));
		%obc.c_eff_cov = -Model.c.alpha*Model.c.alpha'/(alpha^2*(alpha + 1));
    otherwise
        disp('Error: invalid c model type (function OBC).')
end

for i = 1:Params.M
    obc.mu_eff(i, :) = Model.dist.m(i, :);
end

switch Model.dist.structure
    case 'known'
        for i = 1:Params.M
            obc.Sigma_eff{i} = ((Model.dist.nu(i) + 1)/Model.dist.nu(i))*Model.dist.Sigma{i};
        end
    case 'identity'
        switch Model.dist.type
            case 'indep'
                for i = 1:Params.M
                    obc.df(i) = (Model.dist.kappa(i) + Params.d + 1)*Params.d - 2;
                    obc.Sigma_eff{i} = ((Model.dist.nu(i) + 1)/Model.dist.nu(i))*eye(Params.d)*Model.dist.S{i}/obc.df(i);
                    obc.corr{i} = eye(Params.d);
                    obc.stddev{i} = sqrt(((Model.dist.nu(i) + 1)/Model.dist.nu(i))*Model.dist.S{i}*ones(Params.d, 1)/obc.df(i));
                end
            case 'homo'
                for i = 1:Params.M
                    obc.df(i) = (Model.dist.kappa + Params.d + 1)*Params.d - 2;
                    obc.Sigma_eff{i} = ((Model.dist.nu(i) + 1)/Model.dist.nu(i))*eye(Params.d)*Model.dist.S/obc.df(i);
                    obc.corr{i} = eye(Params.d);
                    obc.stddev{i} = sqrt(((Model.dist.nu(i) + 1)/Model.dist.nu(i))*Model.dist.S*ones(Params.d, 1)/obc.df(i));
                end
            otherwise
                disp('Error: invalid covariance model type (function OBC, identity case).')
        end        
    case 'arbitrary'
        switch Model.dist.type
            case 'indep'
                for i = 1:Params.M
                    obc.df(i) = Model.dist.kappa(i) - Params.d + 1;
                    obc.Sigma_eff{i} = ((Model.dist.nu(i) + 1)/Model.dist.nu(i))*Model.dist.S{i}/obc.df(i);
                end
            case 'homo'
                for i = 1:Params.M
                    obc.df(i) = Model.dist.kappa - Params.d + 1;
                    obc.Sigma_eff{i} = ((Model.dist.nu(i) + 1)/Model.dist.nu(i))*Model.dist.S/obc.df(i);
                end
            otherwise
                disp('Error: invalid covariance model type (function OBC, arbitrary case).')
        end
        
        for i = 1:Params.M
            [obc.corr{i}, obc.stddev{i}] = corrcov(obc.Sigma_eff{i});
        end
    otherwise
        disp('Error: invalid covariance model structure (function OBC).')
end