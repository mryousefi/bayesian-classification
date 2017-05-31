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
%========================================================================

function Classifiers = risk_estimation(Params, Model_Prior, Model_Posterior, Classifiers, lambda, x, n, x_tst, n_tst)

% Exact BRE and exact MSE of BRE
if Params.M == 2
    if Params.BRE_EXACT.do
        if Params.LDA.do
            if ~Classifiers.LDA.error
                [Classifiers.LDA.BRE_exact, varepsilon_exact_LDA] = BRE_exact(Params, Model_Posterior, Classifiers.LDA, lambda, Classifiers.OBC);
            else
                Classifiers.LDA.BRE_exact = NaN;
                varepsilon_exact_LDA = NaN;
            end
        end
        
        if Params.LSVM.do
            if ~Classifiers.LSVM.error
                [Classifiers.LSVM.BRE_exact, varepsilon_exact_LSVM] = BRE_exact(Params, Model_Posterior, Classifiers.LSVM, lambda, Classifiers.OBC);
            else
                Classifiers.LSVM.BRE_exact = NaN;
                varepsilon_exact_LSVM = NaN;
            end
        end
    end
    
    if Params.MSE_EXACT.do
        if Params.LDA.do
            if ~Classifiers.LDA.error
                Classifiers.LDA.BRE_MSE_exact = MSE_exact(Params, Model_Posterior, Classifiers.LDA, lambda, Classifiers.OBC, true, varepsilon_exact_LDA);
            else
                Classifiers.LDA.BRE_MSE_exact = NaN;
            end
        end
        
        if Params.LSVM.do
            if ~Classifiers.LSVM.error
                Classifiers.LSVM.BRE_MSE_exact = MSE_exact(Params, Model_Posterior, Classifiers.LSVM, lambda, Classifiers.OBC, true, varepsilon_exact_LSVM);
            else
                Classifiers.LSVM.BRE_MSE_exact = NaN;
            end
        end
    end
end

% Approximate BRE
if Params.BRE_APPROX.do
    [x_synthetic, x_centered] = OBC_generate_data(Params, Model_Posterior, Classifiers.OBC);
    
    if Params.OBC.do
        [Classifiers.OBC.BRE_approx, varepsilon_x_OBC] = BRE_approx(Params, Model_Posterior, Classifiers.OBC, lambda, Classifiers.OBC, x_synthetic);
    end
    
    if Params.LDA.do
        if ~Classifiers.LDA.error
            [Classifiers.LDA.BRE_approx, varepsilon_x_LDA] = BRE_approx(Params, Model_Posterior, Classifiers.LDA, lambda, Classifiers.OBC, x_synthetic);
        else
            Classifiers.LDA.BRE_approx = NaN;
            varepsilon_x_LDA = NaN;
        end
    end
    
    if Params.QDA.do
        if ~Classifiers.QDA.error
            [Classifiers.QDA.BRE_approx, varepsilon_x_QDA] = BRE_approx(Params, Model_Posterior, Classifiers.QDA, lambda, Classifiers.OBC, x_synthetic);
        else
            Classifiers.QDA.BRE_approx = NaN;
            varepsilon_x_QDA = NaN;
        end
    end
    
    if Params.LSVM.do
        if ~Classifiers.LSVM.error
            [Classifiers.LSVM.BRE_approx, varepsilon_x_LSVM] = BRE_approx(Params, Model_Posterior, Classifiers.LSVM, lambda, Classifiers.OBC, x_synthetic);
        else
            Classifiers.LSVM.BRE_approx = NaN;
            varepsilon_x_LSVM = NaN;
        end
    end
    
    if Params.RBFSVM.do
        if ~Classifiers.RBFSVM.error
            [Classifiers.RBFSVM.BRE_approx, varepsilon_x_RBFSVM] = BRE_approx(Params, Model_Posterior, Classifiers.RBFSVM, lambda, Classifiers.OBC, x_synthetic);
        else
            Classifiers.RBFSVM.BRE_approx = NaN;
            varepsilon_x_RBFSVM = NaN;
        end
    end
end

% Approximate MSE of BRE
if Params.MSE_APPROX.do
    w_synthetic = OBC_generate_data_conditional(Params, Model_Posterior, Classifiers.OBC, x_centered);
    
    if Params.OBC.do
        [Classifiers.OBC.BRE_approx2, varepsilon_w_OBC] = BRE_approx(Params, Model_Posterior, Classifiers.OBC, lambda, Classifiers.OBC, w_synthetic);
        Classifiers.OBC.BRE_MSE_approx = MSE_approx(Params, Model_Posterior, Classifiers.OBC, lambda, Classifiers.OBC, x_synthetic, w_synthetic, true, varepsilon_x_OBC, varepsilon_w_OBC);
    end
    
    if Params.LDA.do
        if ~Classifiers.LDA.error
            [Classifiers.LDA.BRE_approx2, varepsilon_w_LDA] = BRE_approx(Params, Model_Posterior, Classifiers.LDA, lambda, Classifiers.OBC, w_synthetic);
            Classifiers.LDA.BRE_MSE_approx = MSE_approx(Params, Model_Posterior, Classifiers.LDA, lambda, Classifiers.OBC, x_synthetic, w_synthetic, true, varepsilon_x_LDA, varepsilon_w_LDA);
        else
            Classifiers.LDA.BRE_approx2 = NaN;
            varepsilon_w_LDA = NaN;
            Classifiers.LDA.BRE_MSE_approx = NaN;
        end
    end
    
    if Params.QDA.do
        if ~Classifiers.QDA.error
            [Classifiers.QDA.BRE_approx2, varepsilon_w_QDA] = BRE_approx(Params, Model_Posterior, Classifiers.QDA, lambda, Classifiers.OBC, w_synthetic);
            Classifiers.QDA.BRE_MSE_approx = MSE_approx(Params, Model_Posterior, Classifiers.QDA, lambda, Classifiers.OBC, x_synthetic, w_synthetic, true, varepsilon_x_QDA, varepsilon_w_QDA);
        else
            Classifiers.QDA.BRE_approx2 = NaN;
            varepsilon_w_QDA = NaN;
            Classifiers.QDA.BRE_MSE_approx = NaN;
        end
    end
    
    if Params.LSVM.do
        if ~Classifiers.LSVM.error
            [Classifiers.LSVM.BRE_approx2, varepsilon_w_LSVM] = BRE_approx(Params, Model_Posterior, Classifiers.LSVM, lambda, Classifiers.OBC, w_synthetic);
            Classifiers.LSVM.BRE_MSE_approx = MSE_approx(Params, Model_Posterior, Classifiers.LSVM, lambda, Classifiers.OBC, x_synthetic, w_synthetic, true, varepsilon_x_LSVM, varepsilon_w_LSVM);
        else
            Classifiers.LSVM.BRE_approx2 = NaN;
            varepsilon_w_LSVM = NaN;
            Classifiers.LSVM.BRE_MSE_approx = NaN;
        end
    end
    
    if Params.RBFSVM.do
        if ~Classifiers.RBFSVM.error
            [Classifiers.RBFSVM.BRE_approx2, varepsilon_w_RBFSVM] = BRE_approx(Params, Model_Posterior, Classifiers.RBFSVM, lambda, Classifiers.OBC, w_synthetic);
            Classifiers.RBFSVM.BRE_MSE_approx = MSE_approx(Params, Model_Posterior, Classifiers.RBFSVM, lambda, Classifiers.OBC, x_synthetic, w_synthetic, true, varepsilon_x_RBFSVM, varepsilon_w_RBFSVM);
        else
            Classifiers.RBFSVM.BRE_approx2 = NaN;
            varepsilon_w_RBFSVM = NaN;
            Classifiers.RBFSVM.BRE_MSE_approx = NaN;
        end
    end
end

% calculate true error
if Params.TRUE.do
    Classifiers = risk_true(Params, Model_Posterior, Classifiers, lambda, x_tst, n_tst);
end

% Classical error estimators
if Params.RESUB.do
    Classifiers = risk_resub(Params, Model_Posterior, Classifiers, lambda, x, n);
end

if Params.CV.do
    Classifiers = risk_cv(Params, Model_Prior, Classifiers, lambda, x, n, 10, 10);
end

if Params.LOO.do
    Classifiers = risk_loo(Params, Model_Prior, Classifiers, lambda, x, n);
end

if Params.BOOT.do
    Classifiers = risk_boot(Params, Model_Prior, Classifiers, lambda, x, n, 100);
end