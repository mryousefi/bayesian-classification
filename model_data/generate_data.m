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

function [n, x, n_tst, x_tst] = generate_data(Params, c, mu, Sigma)

switch Params.train.method
    case 'random'
        n = mnrnd(Params.train.n, c');
        n = n';
        
        n_tst = mnrnd(Params.test.n, c');
        n_tst = n_tst';
    case 'stratified'
        n = floor(Params.train.n*c);
        n_tst = floor(Params.test.n*c);
        while sum(n) ~= Params.train.n
            [~, index] = min(n);
            n(index(1)) = n(index(1)) + 1;
        end
        
        while sum(n_tst) ~= Params.test.n
            [~, index] = min(n_tst);
            n_tst(index(1)) = n_tst(index(1)) + 1;
        end
        
    otherwise
        disp('Error: invalid sampling method (function generate_data).')
end

x = cell(Params.M, 1);
x_tst = cell(Params.M, 1);
for i = 1:Params.M
    x{i} = mvnrnd(mu(i, :), Sigma{i}, n(i));
    x_tst{i} = mvnrnd(mu(i, :), Sigma{i}, n_tst(i));
end
