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

function Model = define_priors(Params, Model)

switch Model.c.type
    case 'known'
        Model.c.value = (1/Params.M)*ones(Params.M, 1);
    case 'dirichlet'
        Model.c.alpha = (Model.c.info/Params.M)*ones(Params.M, 1);    % Model.c.info is the sum of alpha
    otherwise
        display('Error: invalid c model type (function define_priors). Default (''known'') is used.');
        Model.c.type = 'known';
        Model.c.value = (1/Params.M)*ones(Params.M, 1);
end

switch Model.dist.structure
    case 'fixed'
        %% Assume mean and cov are both fixed (code for OBC stuff not done)
        Model.dist.type = 'indep';
        for i = 1:Params.M
            Model.dist.mu(i, :)= Model.dist.mean.value(i, :);
            Model.dist.Sigma{i} = Model.dist.cov.value{i};
        end
    case 'known'
        %% Assume mean is Gaussian, cov is fixed
        Model.dist.type = 'indep';
        for i = 1:Params.M
            Model.dist.nu(i) = Model.dist.mean.info(i);                 % need nu > 0
            Model.dist.m(i, :)= Model.dist.mean.value(i, :);
            Model.dist.Sigma{i} = Model.dist.cov.value{i};
        end
    case 'identity'
        %% Assume mean is Gaussian, cov is scaled identity
        switch Model.dist.type
            case 'indep'
                for i = 1:Params.M
                    Model.dist.nu(i) = Model.dist.mean.info(i);                 % need nu > 0
                    Model.dist.m(i, :)= Model.dist.mean.value(i, :);
                    Model.dist.kappa(i) = Model.dist.cov.info(i);  % need (Model.dist.kappa(i) + Params.d + 1)*Params.d - 2 > 0
                    Model.dist.S{i} = trace(Model.dist.cov.value{i}*((Model.dist.kappa(i) + Params.d + 1)*Params.d - 4));
                end
            case 'homo'
                for i = 1:Params.M
                    Model.dist.nu(i) = Model.dist.mean.info(i);                 % need nu > 0
                    Model.dist.m(i, :)= Model.dist.mean.value(i, :);
                end
                Model.dist.kappa = Model.dist.cov.info(1);         % need (Model.dist.kappa(i) + Params.d + 1)*Params.d - 2 > 0
                Model.dist.S = trace(Model.dist.cov.value{1}*((Model.dist.kappa + Params.d + 1)*Params.d - 4));
            otherwise
                disp('Error: invalid covariance model type (function define_priors, identity case). Default (''indep'') is used.')
                Model.dist.type = 'indep';
                for i = 1:Params.M
                    Model.dist.nu(i) = Model.dist.mean.info(i);                 % need nu > 0
                    Model.dist.m(i, :)= Model.dist.mean.value(i, :);
                    Model.dist.kappa(i) = Model.dist.cov.info(i);  % need (Model.dist.kappa(i) + Params.d + 1)*Params.d - 2 > 0
                    Model.dist.S{i} = trace(Model.dist.cov.value{i}*((Model.dist.kappa(i) + Params.d + 1)*Params.d - 4));
                end
        end
    case 'arbitrary'
        %% Assume mean is Gaussian, cov is arbitrary valid covariance
        switch Model.dist.type
            case 'indep'
                for i = 1:Params.M
                    Model.dist.nu(i) = Model.dist.mean.info(i);                 % need nu > 0
                    Model.dist.m(i, :)= Model.dist.mean.value(i, :);
                    Model.dist.kappa(i) = Model.dist.cov.info(i);    % need kappa - Params.d + 1 > 0
                    Model.dist.S{i} = Model.dist.cov.value{i}*(Model.dist.kappa(i) - Params.d - 1);
                end
            case 'homo'
                for i = 1:Params.M
                    Model.dist.nu(i) = Model.dist.mean.info(i);                 % need nu > 0
                    Model.dist.m(i, :)= Model.dist.mean.value(i, :);
                end
                Model.dist.kappa = Model.dist.cov.info(1);           % need kappa - Params.d + 1 > 0
                Model.dist.S = Model.dist.cov.value{1}*(Model.dist.kappa - Params.d - 1);
            otherwise
                disp('Error: invalid covariance model type (function define_priors, arbitrary case). Default (''indep'') is used.')
                Model.dist.type = 'indep';
                for i = 1:Params.M
                    Model.dist.nu(i) = Model.dist.mean.info(i);                 % need nu > 0
                    Model.dist.m(i, :)= Model.dist.mean.value(i, :);
                    Model.dist.kappa(i) = Model.dist.cov.info(i);    % need kappa - Params.d + 1 > 0
                    Model.dist.S{i} = Model.dist.cov.value{i}*(Model.dist.kappa(i) - Params.d - 1);
                end
        end
    otherwise
        disp('Error: invalid covariance model structure (function define_priors). Default (''known'') is used.')
        Model.dist.structure = 'known';
        Model.dist.type = 'indep';
        for i = 1:Params.M
            Model.dist.nu(i) = Model.dist.mean.info;                 % need nu > 0
            Model.dist.m(i, :)= Model.dist.mean.value(i, :);
            Model.dist.Sigma{i} = Model.dist.cov.value{i};
        end
end

Model = check_valid_model(Params, Model);