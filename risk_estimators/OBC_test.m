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

function labels = OBC_test(Params, Model, obc, lambda, x)

prob = zeros(size(x, 1), Params.M);

switch Model.dist.structure
    case 'known'
        for i = 1:Params.M
            prob(:, i) = obc.c_eff(i)*mvnpdf(x, obc.mu_eff(i, :), obc.Sigma_eff{i});
        end
    case {'identity', 'arbitrary'}
        for i = 1:Params.M
            %% New method
            temp = bsxfun(@minus, x, obc.mu_eff(i, :));
            prob(:, i) = obc.c_eff(i)*(1/prod(obc.stddev{i}))*mvtpdf(temp*diag(1./obc.stddev{i}), obc.corr{i}, obc.df(i));
            % prob(:, i) = obc.c_eff(i)*(1/prod(obc.stddev{i}))*mvtpdf((x - repmat(obc.mu_eff(i, :), size(x, 1), 1))*diag(1./obc.stddev{i}), obc.corr{i}, obc.df(i));
            
            %% Old method - for debugging
            % thing = x-repmat(obc.mu_eff(i, :), size(x, 1), 1);
            % Mahalanobis_all{i} = sum( (thing*((1/obc.df(i))*inv(obc.Sigma_eff{i}))).*thing ,2);
            % K_all{i} = ( gamma((obc.df(i)+Params.d)/2)/( gamma(obc.df(i)/2) * pi^(Params.d/2) * obc.df(i)^(Params.d/2) * det(obc.Sigma_eff{i})^(1/2) ) );
            % prob2(:, i) = obc.c_eff(i) * K_all{i} * (1 + Mahalanobis_all{i}).^(-(obc.df(i) + Params.d)/2);
        end
    otherwise
        disp('Error: invalid covariance model structure (function OBC_test).')
end

g_i = prob*lambda';
[~, labels] = min(g_i, [], 2);

%% Old method - for debugging
% classifier.k0 = Model.dist.kappa(1) + 1;
% classifier.k1 = Model.dist.kappa(2) + 1;
% classifier.Psi0_inv = inv(Model.dist.S{1}*(Model.dist.nu(1) + 1)/Model.dist.nu(1));
% classifier.Psi1_inv = inv(Model.dist.S{2}*(Model.dist.nu(2) + 1)/Model.dist.nu(2));
% classifier.m0 = Model.dist.m(1, :);
% classifier.m1 = Model.dist.m(2, :);
% classifier.K = (Model.c.value(2)/Model.c.value(1))^2*(det(classifier.Psi1_inv)/det(classifier.Psi0_inv)) ...
%     *(beta((classifier.k0-Params.d)/2, Params.d/2)/beta((classifier.k1-Params.d)/2, Params.d/2))^2;
% 
% thing0 = x-repmat(classifier.m0, size(x, 1), 1);
% thing1 = x-repmat(classifier.m1, size(x, 1), 1);
% 
% Mahalanobis0 = sum((thing0*classifier.Psi0_inv).*thing0,2);
% Mahalanobis1 = sum((thing1*classifier.Psi1_inv).*thing1,2);
% 
% % g(x) = K*(1+(x-m0)*inv(Psi0)*(x-m0)^T)^k0 - (1+(x-m1)*inv(Psi1)*(x-m1)^T)^k1
% output0 = classifier.K*(1+Mahalanobis0).^classifier.k0;
% output1 = (1+Mahalanobis1).^classifier.k1;
% output = output0 - output1;
% 
% (output0./output1).^(1/2)
% prob2(:, 2)./prob2(:, 1)
% prob(:, 2)./prob(:, 1)