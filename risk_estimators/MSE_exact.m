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

function risk_mse = MSE_exact(Params, Model, Classifier, lambda, obc, flag_risk_external, varepsilon)

if flag_risk_external == false
    [~, varepsilon] = BRE_exact(Params, Model, Classifier, lambda, obc);
    risk_cond = sum(varepsilon.*lambda, 1);
    risk_est = risk_cond*obc.c_eff;
else
    risk_cond = sum(varepsilon.*lambda, 1);
    risk_est = risk_cond*obc.c_eff;
end

a = Classifier.a{1, 2};
b = Classifier.b{1, 2};

if (strcmp(Classifier.type, 'lda') || strcmp(Classifier.type, 'lsvm')) && (Params.M == 2)
    switch Model.dist.type
        case 'indep'
            term = NaN(Params.M);
            for y = 1:Params.M
                for z = 1:(y-1)
                    term(y, z) = term(z, y);
                end
                for z = y
                    varepsilon_2 = NaN(Params.M);
                    switch Model.dist.structure
                        case 'known'
                            i = 1;
                            m = (-1)^(i)*(obc.mu_eff(y, :)*a + b);
                            c = a'*obc.Sigma_eff{y}*a;
                            rho = 1/(Model.dist.nu(y) + 1);
                            varepsilon_2(i, i) = mvncdf([0, 0], [m, m], [c, rho*c; rho*c, c]);
                        case {'identity', 'arbitrary'}
                            i = 1;
                            k = obc.df(y);
                            m = (-1)^(i)*(obc.mu_eff(y, :)*a + b);
                            c = a'*obc.Sigma_eff{y}*a;
                            rho = 1/(Model.dist.nu(y) + 1);
                            varepsilon_2(i, i) = mvtcdf(-[m, m]/sqrt(c), [1, rho; rho, 1], k);
                        otherwise
                            disp('Error: invalid covariance model structure (function MSE_exact).')
                    end
                    varepsilon_2(i, 3-i) = varepsilon(i, y) - varepsilon_2(i, i);
                    varepsilon_2(3-i, i) = varepsilon_2(i, 3-i);
                    varepsilon_2(3-i, 3-i) = 1 - 2*varepsilon(i, y) + varepsilon_2(i, i);
                    term(y, y) = (lambda(:, y)')*varepsilon_2*(lambda(:, y));
                end
                for z = (y+1):Params.M
                    varepsilon_2 = NaN(Params.M);
                    varepsilon_2(1, 1) = varepsilon(1, y)*varepsilon(1, z);
                    varepsilon_2(1, 2) = varepsilon(1, y)*varepsilon(2, z);
                    varepsilon_2(2, 1) = varepsilon(2, y)*varepsilon(1, z);
                    varepsilon_2(2, 2) = varepsilon(2, y)*varepsilon(2, z);
                    term(y, z) = (lambda(:, y)')*varepsilon_2*(lambda(:, z));
                end
            end
            
            mom2 = sum(sum(obc.c_eff_cross.*term));
            risk_mse = mom2 - risk_est^2;
        case 'homo'
            term = NaN(Params.M);
            for y = 1:Params.M
                for z = 1:(y-1)
                    term(y, z) = term(z, y);
                end
                for z = y
                    varepsilon_2 = NaN(Params.M);
                    switch Model.dist.structure
                        case {'identity', 'arbitrary'}
                            i = 1;
                            k = obc.df(y);
                            m = (-1)^(i)*(obc.mu_eff(y, :)*a + b);
                            c = a'*obc.Sigma_eff{y}*a;
                            rho = 1/(Model.dist.nu(y) + 1);
                            varepsilon_2(i, i) = mvtcdf(-[m, m]/sqrt(c), [1, rho; rho, 1], k);
                        otherwise
                            disp('Error: invalid covariance model structure (function MSE_exact).')
                    end
                    varepsilon_2(i, 3-i) = varepsilon(i, y) - varepsilon_2(i, i);
                    varepsilon_2(3-i, i) = varepsilon_2(i, 3-i);
                    varepsilon_2(3-i, 3-i) = 1 - 2*varepsilon(i, y) + varepsilon_2(i, i);
                    term(y, y) = (lambda(:, y)')*varepsilon_2*(lambda(:, y));
                end
                for z = (y+1):Params.M
                    varepsilon_2 = NaN(Params.M);
                    switch Model.dist.structure
                        case {'identity', 'arbitrary'}
                            i = 1;
                            k = obc.df(y);
                            my = (-1)^(i)*(obc.mu_eff(y, :)*a + b);
                            mz = (-1)^(i)*(obc.mu_eff(z, :)*a + b);
                            cy = a'*obc.Sigma_eff{y}*a;
                            cz = a'*obc.Sigma_eff{z}*a;
                            rho = 0;
                            varepsilon_2(i, i) = mvtcdf(-[my/sqrt(cy), mz/sqrt(cz)], [1, rho; rho, 1], k);
                        otherwise
                            disp('Error: invalid covariance model structure (function MSE_exact).')
                    end
                    varepsilon_2(i, 3-i) = varepsilon(i, y) - varepsilon_2(i, i);
                    varepsilon_2(3-i, i) = varepsilon(i, z) - varepsilon_2(i, i);
                    varepsilon_2(3-i, 3-i) = 1 - varepsilon(i, y) - varepsilon(i, z) + varepsilon_2(i, i);
                    term(y, z) = (lambda(:, y)')*varepsilon_2*(lambda(:, z));
                end
            end
            mom2 = sum(sum(obc.c_eff_cross.*term));
            risk_mse = mom2 - risk_est^2;
        otherwise
            disp('Error: invalid covariance model type (function MSE_exact).')
    end
else
    disp('Error: invalid classifier or model type (function MSE_exact).')
end