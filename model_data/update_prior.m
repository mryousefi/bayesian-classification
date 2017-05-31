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

function Posterior = update_prior(Params, Prior, x, n)

Posterior = Prior;

switch Prior.c.type
    case 'known'
        Posterior.c = Prior.c;
    case 'dirichlet'
        Posterior.c.alpha = Prior.c.alpha + n;
    otherwise
        disp('Error: invalid c prior type (function update_prior, update step).')
end

mu_hat = zeros(Params.M, Params.d);
Sigma_hat = cell(Params.M, 1);
for i = 1:Params.M
    switch n(i)
        case 0
            mu_hat(i, :) = zeros(1, Params.d);
            Sigma_hat{i} = zeros(Params.d);
        otherwise
            mu_hat(i, :) = mean(x{i}, 1);
            Sigma_hat{i} = cov(x{i});
    end
end

for i = 1:Params.M
    Posterior.dist.nu(i) = Prior.dist.nu(i) + n(i);
    Posterior.dist.m(i, :) = (Prior.dist.nu(i)*Prior.dist.m(i, :) + n(i)*mu_hat(i, :))/(Prior.dist.nu(i) + n(i));
end

switch Prior.dist.structure
    case 'known'
        % for i = 1:Params.M
        %     Posterior.dist.Sigma{i} = Prior.dist.Sigma{i} ;
        % end
    case 'identity'
        switch Prior.dist.type
            case 'indep'
                for i = 1:Params.M
                    Posterior.dist.kappa(i) = Prior.dist.kappa(i) + n(i);
                    Posterior.dist.S{i} = Prior.dist.S{i} + trace((n(i) - 1)*Sigma_hat{i} ...
                        + (Prior.dist.nu(i)*n(i))/(Prior.dist.nu(i) + n(i))*(mu_hat(i, :) - Prior.dist.m(i, :))'*(mu_hat(i, :) - Prior.dist.m(i, :)));
                end
            case 'homo'
                Posterior.dist.kappa = Prior.dist.kappa + sum(n);
                temp_S = zeros(Params.d);
                for i = 1:Params.M
                    temp_S = temp_S + (n(i) - 1)*Sigma_hat{i} ...
                        + (Prior.dist.nu(i)*n(i))/(Prior.dist.nu(i) + n(i))*(mu_hat(i, :) - Prior.dist.m(i, :))'*(mu_hat(i, :) - Prior.dist.m(i, :));
                end
                Posterior.dist.S = Prior.dist.S + trace(temp_S);
            otherwise
                disp('Error: invalid covariance prior type (function update_prior, identity case).')
        end
    case 'arbitrary'
        switch Prior.dist.type
            case 'indep'
                for i = 1:Params.M
                    Posterior.dist.kappa(i) = Prior.dist.kappa(i) + n(i);
                    Posterior.dist.S{i} = Prior.dist.S{i} + (n(i) - 1)*Sigma_hat{i} ...
                        + (Prior.dist.nu(i)*n(i))/(Prior.dist.nu(i) + n(i))*(mu_hat(i, :) - Prior.dist.m(i, :))'*(mu_hat(i, :) - Prior.dist.m(i, :));
                end
            case 'homo'
                Posterior.dist.kappa = Prior.dist.kappa + sum(n);
                temp_S = zeros(Params.d);
                for i = 1:Params.M
                    temp_S = temp_S + (n(i) - 1)*Sigma_hat{i} ...
                        + (Prior.dist.nu(i)*n(i))/(Prior.dist.nu(i) + n(i))*(mu_hat(i, :) - Prior.dist.m(i, :))'*(mu_hat(i, :) - Prior.dist.m(i, :));
                end
                Posterior.dist.S = Prior.dist.S + temp_S;
            otherwise
                disp('Error: invalid covariance prior type (function update_prior, arbitrary case).')
        end
    otherwise
        disp('Error: invalid covariance prior structure (function update_prior).')
end

switch Prior.c.type
    case 'known'
        if ~all(Prior.dist.valid)
            Posterior = check_valid_model(Params, Posterior);
        end
    case 'dirichlet'
        if ~(Prior.c.valid && all(Prior.dist.valid))
            Posterior = check_valid_model(Params, Posterior);
        end
    otherwise
        disp('Error: invalid c prior type (function update_prior, check validity).')
end