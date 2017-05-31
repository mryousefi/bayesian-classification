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

function w = OBC_generate_data_conditional(Params, Model, obc, x_centered)

switch Model.dist.structure
    case 'known'
        mu_eff_w_same = cell(Params.M, 1);
        for y = 1:Params.M
            mu_eff_w_same{y} = repmat(Model.dist.m(y, :), Params.synthetic.n(y), 1) + (Model.dist.nu(y) + 1)^(-1)*x_centered{y};
        end
        
        w = cell(Params.M, 1);
        for y = 1:Params.M
            Sigma_eff_w = ((Model.dist.nu(y) + 2)/(Model.dist.nu(y) + 1))*Model.dist.Sigma{y};
            w{y} = mvnrnd(mu_eff_w_same{y}, Sigma_eff_w, Params.synthetic.n(y));
        end
    case 'identity'
        switch Model.dist.type
            case 'indep'
                mu_eff_w_same = cell(Params.M, 1);
                for y = 1:Params.M
                    mu_eff_w_same{y} = repmat(Model.dist.m(y, :), Params.synthetic.n(y), 1) + (Model.dist.nu(y) + 1)^(-1)*x_centered{y};
                end
                
                w = cell(Params.M, 1);
                for y = 1:Params.M
                    df_w = (Model.dist.kappa(y) + Params.d + 2)*Params.d - 2;
                    S_starstar = Model.dist.S{y} + (Model.dist.nu(y)/(Model.dist.nu(y)+1))*sum(x_centered{y}.^2, 2);
                    stddev_w = sqrt(((Model.dist.nu(y) + 2)/(Model.dist.nu(y) + 1))*S_starstar/df_w);
                    w{y} = generate_w_identity(Params, df_w, mu_eff_w_same{y}, stddev_w, Params.synthetic.n(y));
                end
            case 'homo'
                %% The following code find the joint density, but in this funtion we want to conditional density
                % x_synthetic = cell(Params.M);
                % w_synthetic = cell(Params.M);
                % obc = Classifiers.OBC;
                % D = Params.d;
                % for y = 1:Params.M
                %     rho = 1/(Model.dist.nu(y) + 1);
                %     thing = mvtrnd([obc.corr{y}, rho*obc.corr{y}; rho*obc.corr{y}, obc.corr{y}], ...
                %         obc.df(y), ...
                %         Params.synthetic.n(y))*diag([obc.stddev{y}; obc.stddev{y}]);
                %     x_synthetic{y, y} = thing(:, 1:D) + repmat(obc.mu_eff(y, :), Params.synthetic.n(y), 1);
                %     w_synthetic{y, y} = thing(:, (D+1):end) + repmat(obc.mu_eff(y, :), Params.synthetic.n(y), 1);
                %
                %     for z = [1:(y-1), (y+1):Params.M]
                %         thing = mvtrnd([obc.corr{y}, zeros(D); zeros(D), obc.corr{z}], ...
                %             obc.df(y), ...
                %             Params.synthetic.n(y))*diag([obc.stddev{y}; obc.stddev{z}]);
                %         x_synthetic{y, z} = thing(:, 1:D) + repmat(obc.mu_eff(y, :), Params.synthetic.n(y), 1);
                %         w_synthetic{z, y} = thing(:, (D+1):end) + repmat(obc.mu_eff(z, :), Params.synthetic.n(y), 1);
                %     end
                % end
                
                mu_eff_w_same = cell(Params.M, 1);
                for y = 1:Params.M
                    mu_eff_w_same{y} = repmat(Model.dist.m(y, :), Params.synthetic.n(y), 1) + (Model.dist.nu(y) + 1)^(-1)*x_centered{y, y};
                end
                
                mu_eff_w_diff = cell(Params.M, 1);
                for z = 1:Params.M
                    mu_eff_w_diff{z} = repmat(Model.dist.m(z, :), Params.synthetic.n(z), 1);
                end
                
                w = cell(Params.M);
                df_w = (Model.dist.kappa + Params.d + 2)*Params.d - 2;
                for y = 1:Params.M
                    S_starstar = Model.dist.S + (Model.dist.nu(y)/(Model.dist.nu(y)+1))*sum(x_centered{y, y}.^2, 2);
                    stddev_w_same = sqrt(((Model.dist.nu(y) + 2)/(Model.dist.nu(y) + 1))*S_starstar/df_w);
                    w{y, y} = generate_w_identity(Params, df_w, mu_eff_w_same{y}, stddev_w_same, Params.synthetic.n(y));
                    
                    for z = [1:(y - 1), (y + 1):Params.M]
                        S_starstar = Model.dist.S + (Model.dist.nu(y)/(Model.dist.nu(y)+1))*sum(x_centered{y, z}.^2, 2);
                        stddev_w_diff = sqrt(((Model.dist.nu(z) + 1)/(Model.dist.nu(z)))*S_starstar/df_w);
                        w{z, y} = generate_w_identity(Params, df_w, mu_eff_w_diff{z}, stddev_w_diff, Params.synthetic.n(z));
                    end
                end
            otherwise
                disp('Error: invalid covariance model type (function OBC_generate_data_conditional, identity case).')
        end
    case 'arbitrary'
        switch Model.dist.type
            case 'indep'
                mu_eff_w_same = cell(Params.M, 1);
                for y = 1:Params.M
                    mu_eff_w_same{y} = repmat(Model.dist.m(y, :), Params.synthetic.n(y), 1) + (Model.dist.nu(y) + 1)^(-1)*x_centered{y};
                end
                
                w = cell(Params.M, 1);
                for y = 1:Params.M
                    df_w = Model.dist.kappa(y) - Params.d + 2;
                    coef_w = ((Model.dist.nu(y) + 2)/(Model.dist.nu(y) + 1))/df_w;
                    R = chol(Model.dist.S{y});
                    x_mod = sqrt((Model.dist.nu(y)/(Model.dist.nu(y)+1)))*x_centered{y};
                    w{y} = generate_w_arbitrary(Params, df_w, mu_eff_w_same{y}, coef_w, R, x_mod, Params.synthetic.n(y));
                end
            case 'homo'
                mu_eff_w_same = cell(Params.M, 1);
                for y = 1:Params.M
                    mu_eff_w_same{y} = repmat(Model.dist.m(y, :), Params.synthetic.n(y), 1) + (Model.dist.nu(y) + 1)^(-1)*x_centered{y, y};
                end
                
                mu_eff_w_diff = cell(Params.M, 1);
                for z = 1:Params.M
                    mu_eff_w_diff{z} = repmat(Model.dist.m(z, :), Params.synthetic.n(z), 1);
                end
                
                w = cell(Params.M);
                df_w = Model.dist.kappa - Params.d + 2;
                R = chol(Model.dist.S);
                for y = 1:Params.M
                    coef_w_same = ((Model.dist.nu(y) + 2)/(Model.dist.nu(y) + 1))/df_w;
                    x_mod = sqrt((Model.dist.nu(y)/(Model.dist.nu(y)+1)))*x_centered{y, y};
                    w{y, y} = generate_w_arbitrary(Params, df_w, mu_eff_w_same{y}, coef_w_same, R, x_mod, Params.synthetic.n(y));
                    
                    for z = [1:(y - 1), (y + 1):Params.M]
                        coef_w_diff = ((Model.dist.nu(z) + 1)/(Model.dist.nu(z)))/df_w;
                        x_mod = sqrt((Model.dist.nu(y)/(Model.dist.nu(y)+1)))*x_centered{y, z};
                        w{z, y} = generate_w_arbitrary(Params, df_w, mu_eff_w_diff{z}, coef_w_diff, R, x_mod, Params.synthetic.n(z));
                    end
                end
            otherwise
                disp('Error: invalid covariance model type (function OBC_generate_data_conditional, arbitrary case).')
        end
    otherwise
        disp('Error: invalid covariance model structure (function OBC_generate_data_conditional).')
end