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

function output = summarize_results(Params, Classifiers, input, nt, irep)

output = input;

% Exact BRE and exact MSE of BRE
if Params.M == 2
    if Params.BRE_EXACT.do
        if Params.LDA.do
            output.LDA.BRE_exact(nt, irep) = [Classifiers.LDA.BRE_exact];
        end
        
        if Params.LSVM.do
            output.LSVM.BRE_exact(nt, irep) = [Classifiers.LSVM.BRE_exact];
        end
    end
    
    if Params.MSE_EXACT.do
        if Params.LDA.do
            output.LDA.BRE_MSE_exact(nt, irep) = [Classifiers.LDA.BRE_MSE_exact];
        end
        
        if Params.LSVM.do
            output.LSVM.BRE_MSE_exact(nt, irep) = [Classifiers.LSVM.BRE_MSE_exact];
        end
    end
end

% Approximate BRE
if Params.BRE_APPROX.do
    
    if Params.OBC.do
        output.OBC.BRE_approx(nt, irep) = [Classifiers.OBC.BRE_approx];
    end
    
    if Params.LDA.do
        output.LDA.BRE_approx(nt, irep) = [Classifiers.LDA.BRE_approx];
    end
    
    if Params.QDA.do
        output.QDA.BRE_approx(nt, irep) = [Classifiers.QDA.BRE_approx];
    end
    
    if Params.LSVM.do
        output.LSVM.BRE_approx(nt, irep) = [Classifiers.LSVM.BRE_approx];
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.BRE_approx(nt, irep) = [Classifiers.RBFSVM.BRE_approx];
    end
end

% Approximate MSE of BRE
if Params.MSE_APPROX.do
    
    if Params.OBC.do
        output.OBC.BRE_approx2(nt, irep) = [Classifiers.OBC.BRE_approx2];
        output.OBC.BRE_MSE_approx(nt, irep) = [Classifiers.OBC.BRE_MSE_approx];
    end
    
    if Params.LDA.do
        output.LDA.BRE_approx2(nt, irep) = [Classifiers.LDA.BRE_approx2];
        output.LDA.BRE_MSE_approx(nt, irep) = [Classifiers.LDA.BRE_MSE_approx];
    end
    
    if Params.QDA.do
        output.QDA.BRE_approx2(nt, irep) = [Classifiers.QDA.BRE_approx2];
        output.QDA.BRE_MSE_approx(nt, irep) = [Classifiers.QDA.BRE_MSE_approx];
    end
    
    if Params.LSVM.do
        output.LSVM.BRE_approx2(nt, irep) = [Classifiers.LSVM.BRE_approx2];
        output.LSVM.BRE_MSE_approx(nt, irep) = [Classifiers.LSVM.BRE_MSE_approx];
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.BRE_approx2(nt, irep) = [Classifiers.RBFSVM.BRE_approx2];
        output.RBFSVM.BRE_MSE_approx(nt, irep) = [Classifiers.RBFSVM.BRE_MSE_approx];
    end
end

% calculate true error
if Params.TRUE.do
    
    if Params.OBC.do
        output.OBC.true(nt, irep) = [Classifiers.OBC.true];
    end
    
    if Params.LDA.do
        output.LDA.true(nt, irep) = [Classifiers.LDA.true];
    end
    
    if Params.QDA.do
        output.QDA.true(nt, irep) = [Classifiers.QDA.true];
    end
    
    if Params.LSVM.do
        output.LSVM.true(nt, irep) = [Classifiers.LSVM.true];
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.true(nt, irep) = [Classifiers.RBFSVM.true];
    end
end

% Classical error estimators
if Params.RESUB.do
    
    if Params.OBC.do
        output.OBC.resub(nt, irep) = [Classifiers.OBC.resub];
    end
    
    if Params.LDA.do
        output.LDA.resub(nt, irep) = [Classifiers.LDA.resub];
    end
    
    if Params.QDA.do
        output.QDA.resub(nt, irep) = [Classifiers.QDA.resub];
    end
    
    if Params.LSVM.do
        output.LSVM.resub(nt, irep) = [Classifiers.LSVM.resub];
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.resub(nt, irep) = [Classifiers.RBFSVM.resub];
    end
end

if Params.CV.do
    
    if Params.OBC.do
        output.OBC.cv(nt, irep) = [Classifiers.OBC.cv];
    end
    
    if Params.LDA.do
        output.LDA.cv(nt, irep) = [Classifiers.LDA.cv];
    end
    
    if Params.QDA.do
        output.QDA.cv(nt, irep) = [Classifiers.QDA.cv];
    end
    
    if Params.LSVM.do
        output.LSVM.cv(nt, irep) = [Classifiers.LSVM.cv];
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.cv(nt, irep) = [Classifiers.RBFSVM.cv];
    end
end

if Params.LOO.do
    
    if Params.OBC.do
        output.OBC.loo(nt, irep) = [Classifiers.OBC.loo];
    end
    
    if Params.LDA.do
        output.LDA.loo(nt, irep) = [Classifiers.LDA.loo];
    end
    
    if Params.QDA.do
        output.QDA.loo(nt, irep) = [Classifiers.QDA.loo];
    end
    
    if Params.LSVM.do
        output.LSVM.loo(nt, irep) = [Classifiers.LSVM.loo];
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.loo(nt, irep) = [Classifiers.RBFSVM.loo];
    end
end

if Params.BOOT.do
    
    if Params.OBC.do
        output.OBC.boot(nt, irep) = [Classifiers.OBC.boot];
    end
    
    if Params.LDA.do
        output.LDA.boot(nt, irep) = [Classifiers.LDA.boot];
    end
    
    if Params.QDA.do
        output.QDA.boot(nt, irep) = [Classifiers.QDA.boot];
    end
    
    if Params.LSVM.do
        output.LSVM.boot(nt, irep) = [Classifiers.LSVM.boot];
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.boot(nt, irep) = [Classifiers.RBFSVM.boot];
    end
    
    
    if Params.OBC.do
        output.OBC.boot632(nt, irep) = [Classifiers.OBC.boot632];
    end
    
    if Params.LDA.do
        output.LDA.boot632(nt, irep) = [Classifiers.LDA.boot632];
    end
    
    if Params.QDA.do
        output.QDA.boot632(nt, irep) = [Classifiers.QDA.boot632];
    end
    
    if Params.LSVM.do
        output.LSVM.boot632(nt, irep) = [Classifiers.LSVM.boot632];
    end
    
    if Params.RBFSVM.do
        output.RBFSVM.boot632(nt, irep) = [Classifiers.RBFSVM.boot632];
    end
end