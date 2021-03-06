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

function [x, x_centered] = OBC_generate_data(Params, Model, obc)

switch Model.dist.structure
    case 'known'
        x_centered = cell(Params.M, 1);
        x = cell(Params.M, 1);
        
        for y = 1:Params.M
            x_centered{y} = mvnrnd(zeros(1, Params.d), obc.Sigma_eff{y}, Params.synthetic.n(y));
            x{y} = x_centered{y} + repmat(obc.mu_eff(y, :), Params.synthetic.n(y), 1);
        end
    case {'identity', 'arbitrary'}
        switch Model.dist.type
            case 'indep'
                x_centered = cell(Params.M, 1);
                x = cell(Params.M, 1);
                
                for y = 1:Params.M
                    x_centered{y} = mvtrnd(obc.corr{y}, obc.df(y), Params.synthetic.n(y))*diag(obc.stddev{y});
                    x{y} = x_centered{y} + repmat(obc.mu_eff(y, :), Params.synthetic.n(y), 1);
                end
            case 'homo'
                x_centered = cell(Params.M);
                x = cell(Params.M);
                
                for y = 1:Params.M
                    for z = 1:Params.M
                        x_centered{y, z} = mvtrnd(obc.corr{y}, obc.df(y), Params.synthetic.n(y))*diag(obc.stddev{y});
                        x{y, z} = x_centered{y, z} + repmat(obc.mu_eff(y, :), Params.synthetic.n(y), 1);
                    end
                end
            otherwise
                disp('Error: invalid covariance model type (function OBC_generate_data).')
        end
    otherwise
        disp('Error: invalid covariance model structure (function OBC_generate_data).')
end

end