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

function output = initialize_results(Params, no_sample_size, internal_rep)

if Params.M == 2
    if Params.BRE_EXACT.do
        if Params.LDA.do
            output.LDA.BRE_exact = NaN(no_sample_size, internal_rep);
        end
        
        if Params.LSVM.do
            output.LSVM.BRE_exact = NaN(no_sample_size, internal_rep);
        end
    end
    
    if Params.MSE_EXACT.do
        if Params.LDA.do
            output.LDA.BRE_MSE_exact = NaN(no_sample_size, internal_rep);
        end
        
        if Params.LSVM.do
            output.LSVM.BRE_MSE_exact = NaN(no_sample_size, internal_rep);
        end
    end
end

% Approximate BRE
if Params.BRE_APPROX.do
    
    if Params.OBC.do
        output.OBC.BRE_approx = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LDA.do
        output.LDA.BRE_approx = NaN(no_sample_size, internal_rep);
    end
    
    if Params.QDA.do
        output.QDA.BRE_approx = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LSVM.do
        output.LSVM.BRE_approx = NaN(no_sample_size, internal_rep);
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.BRE_approx = NaN(no_sample_size, internal_rep);
    end
end

% Approximate MSE of BRE
if Params.MSE_APPROX.do
    
    if Params.OBC.do
        output.OBC.BRE_approx2 = NaN(no_sample_size, internal_rep);
        output.OBC.BRE_MSE_approx = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LDA.do
        output.LDA.BRE_approx2 = NaN(no_sample_size, internal_rep);
        output.LDA.BRE_MSE_approx = NaN(no_sample_size, internal_rep);
    end
    
    if Params.QDA.do
        output.QDA.BRE_approx2 = NaN(no_sample_size, internal_rep);
        output.QDA.BRE_MSE_approx = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LSVM.do
        output.LSVM.BRE_approx2 = NaN(no_sample_size, internal_rep);
        output.LSVM.BRE_MSE_approx = NaN(no_sample_size, internal_rep);
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.BRE_approx2 = NaN(no_sample_size, internal_rep);
        output.RBFSVM.BRE_MSE_approx = NaN(no_sample_size, internal_rep);
    end
end

% calculate true error
if Params.TRUE.do
    
    if Params.OBC.do
        output.OBC.true = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LDA.do
        output.LDA.true = NaN(no_sample_size, internal_rep);
    end
    
    if Params.QDA.do
        output.QDA.true = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LSVM.do
        output.LSVM.true = NaN(no_sample_size, internal_rep);
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.true = NaN(no_sample_size, internal_rep);
    end
end

% Classical error estimators
if Params.RESUB.do
    
    if Params.OBC.do
        output.OBC.resub = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LDA.do
        output.LDA.resub = NaN(no_sample_size, internal_rep);
    end
    
    if Params.QDA.do
        output.QDA.resub = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LSVM.do
        output.LSVM.resub = NaN(no_sample_size, internal_rep);
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.resub = NaN(no_sample_size, internal_rep);
    end
end

if Params.CV.do
    
    if Params.OBC.do
        output.OBC.cv = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LDA.do
        output.LDA.cv = NaN(no_sample_size, internal_rep);
    end
    
    if Params.QDA.do
        output.QDA.cv = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LSVM.do
        output.LSVM.cv = NaN(no_sample_size, internal_rep);
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.cv = NaN(no_sample_size, internal_rep);
    end
end

if Params.LOO.do
    
    if Params.OBC.do
        output.OBC.loo = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LDA.do
        output.LDA.loo = NaN(no_sample_size, internal_rep);
    end
    
    if Params.QDA.do
        output.QDA.loo = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LSVM.do
        output.LSVM.loo = NaN(no_sample_size, internal_rep);
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.loo = NaN(no_sample_size, internal_rep);
    end
end

if Params.BOOT.do
    
    if Params.OBC.do
        output.OBC.boot = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LDA.do
        output.LDA.boot = NaN(no_sample_size, internal_rep);
    end
    
    if Params.QDA.do
        output.QDA.boot = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LSVM.do
        output.LSVM.boot = NaN(no_sample_size, internal_rep);
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.boot = NaN(no_sample_size, internal_rep);
    end
    
    
    if Params.OBC.do
        output.OBC.boot632 = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LDA.do
        output.LDA.boot632 = NaN(no_sample_size, internal_rep);
    end
    
    if Params.QDA.do
        output.QDA.boot632 = NaN(no_sample_size, internal_rep);
    end
    
    if Params.LSVM.do
        output.LSVM.boot632 = NaN(no_sample_size, internal_rep);
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.boot632 = NaN(no_sample_size, internal_rep);
    end
end