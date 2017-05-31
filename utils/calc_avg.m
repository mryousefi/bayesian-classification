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

function output = calc_avg(Params, Classifiers, sample_size, internal_rep)

% Exact BRE and exact MSE of BRE
if Params.M == 2
    if Params.BRE_EXACT.do
        if Params.LDA.do
            temp = zeros(length(sample_size), internal_rep);
            for nt = 1:length(sample_size)
                for irep = 1:internal_rep
                    temp(nt, irep) = Classifiers{nt, irep}.LDA.BRE_exact;
                end
            end
            output.LDA.BRE_exact = mean(temp, 2);
        end
        
        if Params.LSVM.do
            temp = zeros(length(sample_size), internal_rep);
            for nt = 1:length(sample_size)
                for irep = 1:internal_rep
                    temp(nt, irep) = Classifiers{nt, irep}.LSVM.BRE_exact;
                end
            end
            output.LSVM.BRE_exact = mean(temp, 2);
        end
    end
    
    if Params.MSE_EXACT.do
        if Params.LDA.do
            temp = zeros(length(sample_size), internal_rep);
            for nt = 1:length(sample_size)
                for irep = 1:internal_rep
                    temp(nt, irep) = Classifiers{nt, irep}.LDA.BRE_MSE_exact;
                end
            end
            output.LDA.BRE_MSE_exact = mean(temp, 2);
        end
        
        if Params.LSVM.do
            temp = zeros(length(sample_size), internal_rep);
            for nt = 1:length(sample_size)
                for irep = 1:internal_rep
                    temp(nt, irep) = Classifiers{nt, irep}.LSVM.BRE_MSE_exact;
                end
            end
            output.LSVM.BRE_MSE_exact = mean(temp, 2);
        end
    end
end

% Approximate BRE
if Params.BRE_APPROX.do
    
    if Params.OBC.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.OBC.BRE_approx;
            end
        end
        output.OBC.BRE_approx = mean(temp, 2);
    end
    
    if Params.LDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LDA.BRE_approx;
            end
        end
        output.LDA.BRE_approx = mean(temp, 2);
    end
    
    if Params.QDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.QDA.BRE_approx;
            end
        end
        output.QDA.BRE_approx = mean(temp, 2);
    end
    
    if Params.LSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LSVM.BRE_approx;
            end
        end
        output.LSVM.BRE_approx = mean(temp, 2);
    end
    
    if Params.RBFSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.RBFSVM.BRE_approx;
            end
        end
        output.RBFSVM.BRE_approx = mean(temp, 2);
    end
end

% Approximate MSE of BRE
if Params.MSE_APPROX.do
    
    if Params.OBC.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.OBC.BRE_approx2;
            end
        end
        output.OBC.BRE_approx2 = mean(temp, 2);
        
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.OBC.BRE_MSE_approx;
            end
        end
        output.OBC.BRE_MSE_approx = mean(temp, 2);
    end
    
    if Params.LDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LDA.BRE_approx2;
            end
        end
        output.LDA.BRE_approx2 = mean(temp, 2);
        
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LDA.BRE_MSE_approx;
            end
        end
        output.LDA.BRE_MSE_approx = mean(temp, 2);
    end
    
    if Params.QDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.QDA.BRE_approx2;
            end
        end
        output.QDA.BRE_approx2 = mean(temp, 2);
        
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.QDA.BRE_MSE_approx;
            end
        end
        output.QDA.BRE_MSE_approx = mean(temp, 2);
    end
    
    if Params.LSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LSVM.BRE_approx2;
            end
        end
        output.LSVM.BRE_approx2 = mean(temp, 2);
        
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LSVM.BRE_MSE_approx;
            end
        end
        output.LSVM.BRE_MSE_approx = mean(temp, 2);
    end
    
    if Params.RBFSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.RBFSVM.BRE_approx2;
            end
        end
        output.RBFSVM.BRE_approx2 = mean(temp, 2);
        
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.RBFSVM.BRE_MSE_approx;
            end
        end
        output.RBFSVM.BRE_MSE_approx = mean(temp, 2);
    end
end

% calculate true error
if Params.TRUE.do
    
    if Params.OBC.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.OBC.true;
            end
        end
        output.OBC.true = mean(temp, 2);
    end
    
    if Params.LDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LDA.true;
            end
        end
        output.LDA.true = mean(temp, 2);
    end
    
    if Params.QDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.QDA.true;
            end
        end
        output.QDA.true = mean(temp, 2);
    end
    
    if Params.LSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LSVM.true;
            end
        end
        output.LSVM.true = mean(temp, 2);
    end
    
    if Params.RBFSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.RBFSVM.true;
            end
        end
        output.RBFSVM.true = mean(temp, 2);
    end
end

% Classical error estimators
if Params.RESUB.do
    
    if Params.OBC.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.OBC.resub;
            end
        end
        output.OBC.resub = mean(temp, 2);
    end
    
    if Params.LDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LDA.resub;
            end
        end
        output.LDA.resub = mean(temp, 2);
    end
    
    if Params.QDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.QDA.resub;
            end
        end
        output.QDA.resub = mean(temp, 2);
    end
    
    if Params.LSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LSVM.resub;
            end
        end
        output.LSVM.resub = mean(temp, 2);
    end
    
    if Params.RBFSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.RBFSVM.resub;
            end
        end
        output.RBFSVM.resub = mean(temp, 2);
    end
end

if Params.CV.do
    
    if Params.OBC.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.OBC.cv;
            end
        end
        output.OBC.cv = mean(temp, 2);
    end
    
    if Params.LDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LDA.cv;
            end
        end
        output.LDA.cv = mean(temp, 2);
    end
    
    if Params.QDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.QDA.cv;
            end
        end
        output.QDA.cv = mean(temp, 2);
    end
    
    if Params.LSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LSVM.cv;
            end
        end
        output.LSVM.cv = mean(temp, 2);
    end
    
    if Params.RBFSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.RBFSVM.cv;
            end
        end
        output.RBFSVM.cv = mean(temp, 2);
    end
end

if Params.LOO.do
    
    if Params.OBC.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.OBC.loo;
            end
        end
        output.OBC.loo = mean(temp, 2);
    end
    
    if Params.LDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LDA.loo;
            end
        end
        output.LDA.loo = mean(temp, 2);
    end
    
    if Params.QDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.QDA.loo;
            end
        end
        output.QDA.loo = mean(temp, 2);
    end
    
    if Params.LSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LSVM.loo;
            end
        end
        output.LSVM.loo = mean(temp, 2);
    end
    
    if Params.RBFSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.RBFSVM.loo;
            end
        end
        output.RBFSVM.loo = mean(temp, 2);
    end
end

if Params.BOOT.do
    
    if Params.OBC.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.OBC.boot;
            end
        end
        output.OBC.boot = mean(temp, 2);
    end
    
    if Params.LDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LDA.boot;
            end
        end
        output.LDA.boot = mean(temp, 2);
    end
    
    if Params.QDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.QDA.boot;
            end
        end
        output.QDA.boot = mean(temp, 2);
    end
    
    if Params.LSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LSVM.boot;
            end
        end
        output.LSVM.boot = mean(temp, 2);
    end
    
    if Params.RBFSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.RBFSVM.boot;
            end
        end
        output.RBFSVM.boot = mean(temp, 2);
    end
    
    
    if Params.OBC.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.OBC.boot632;
            end
        end
        output.OBC.boot632 = mean(temp, 2);
    end
    
    if Params.LDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LDA.boot632;
            end
        end
        output.LDA.boot632 = mean(temp, 2);
    end
    
    if Params.QDA.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.QDA.boot632;
            end
        end
        output.QDA.boot632 = mean(temp, 2);
    end
    
    if Params.LSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.LSVM.boot632;
            end
        end
        output.LSVM.boot632 = mean(temp, 2);
    end
    
    if Params.RBFSVM.do
        temp = zeros(length(sample_size), internal_rep);
        for nt = 1:length(sample_size)
            for irep = 1:internal_rep
                temp(nt, irep) = Classifiers{nt, irep}.RBFSVM.boot632;
            end
        end
        output.RBFSVM.boot632 = mean(temp, 2);
    end
end