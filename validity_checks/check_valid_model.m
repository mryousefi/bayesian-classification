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

function Model = check_valid_model(Params, Model)

% check if the prior/posterior is proper
switch Model.c.type
    case 'known'
    case 'dirichlet'
        for i = 1:Params.M
            if Model.c.alpha > 0
                Model.c.valid = true;
            else
                Model.c.valid = false;
            end
        end
    otherwise
        disp('Error: invalid c model type (function check_valid_model).')
end

switch Model.dist.structure
    case 'fixed'
    case {'known', 'identity', 'arbitrary'}
        for i = 1:Params.M
            if Model.dist.nu(i) > 0
                Model.dist.valid(i) = true;
            else
                Model.dist.valid(i) = false;
            end
        end
    otherwise
        disp('Error: invalid covariance prior structure (function check_valid_model).')
end

switch Model.dist.structure
    case 'fixed'
    case 'known'
    case 'identity'
        switch Model.dist.type
            case 'indep'
                for i = 1:Params.M
                    if Model.dist.valid(i) == true
                        if (Model.dist.kappa(i) + Params.d + 1)*Params.d - 2 > 0
                            Model.dist.valid(i) = Model.dist.S{i} > 0;
                        else
                            Model.dist.valid(i) = false;
                        end
                    end
                end
            case 'homo'
                if (Model.dist.kappa + Params.d + 1)*Params.d - 2 > 0
                    valid = Model.dist.S > 0;
                else
                    valid = false;
                end
                for i = 1:Params.M
                    Model.dist.valid(i) = Model.dist.valid(i) & valid;
                end
            otherwise
                disp('Error: invalid covariance prior type (function check_valid_model, identity case).')
        end
    case 'arbitrary'
        switch Model.dist.type
            case 'indep'
                for i = 1:Params.M
                    if Model.dist.valid(i) == true
                        if Model.dist.kappa(i) - Params.d + 1 > 0
                            Model.dist.valid(i) = check_valid_covariance(Model.dist.S{i}, Params.d, Model.dist.valid(i));
                        else
                            Model.dist.valid(i) = false;
                        end
                    end
                end
            case 'homo'
                if Model.dist.kappa - Params.d + 1 > 0
                    valid = check_valid_covariance(Model.dist.S, Params.d, true);
                else
                    valid = false;
                end
                for i = 1:Params.M
                    Model.dist.valid(i) = Model.dist.valid(i) & valid;
                end
            otherwise
                disp('Error: invalid covariance prior type (function check_valid_model, arbitrary case).')
        end
    otherwise
        disp('Error: invalid covariance prior structure (function check_valid_model).')
end