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

function report_summary_to_screen(Params, Model, Classifier)

name = upper(Classifier.type);

if Params.RESUB.do
    disp([name, ' resub =          ', num2str(Classifier.resub)]);
end

if Params.LOO.do
    disp([name, ' loo =            ', num2str(Classifier.loo)]);
end

if Params.CV.do
    disp([name, ' cv =             ', num2str(Classifier.cv)]);
end

if Params.BOOT.do
    disp([name, ' boot =           ', num2str(Classifier.boot)]);
    disp([name, ' boot .632 =      ', num2str(Classifier.boot632)]);
end

if Params.BRE_EXACT.do
    if (strcmp(Classifier.type, 'lda') || strcmp(Classifier.type, 'lsvm')) && (Params.M == 2)
        disp([name, ' BRE exact =      ', num2str(Classifier.BRE_exact)]);
    end
end

if Params.MSE_EXACT.do
    if (strcmp(Classifier.type, 'lda') || strcmp(Classifier.type, 'lsvm')) && (Params.M == 2)
        disp([name, ' BRE RMS exact =  ', num2str(sqrt(Classifier.BRE_MSE_exact))]);
    end
end

if Params.BRE_APPROX.do
    disp([name, ' BRE approx x =   ', num2str(Classifier.BRE_approx)]);
end

if Params.MSE_APPROX.do
    disp([name, ' BRE approx w =   ', num2str(Classifier.BRE_approx2)]);
    disp([name, ' BRE RMS approx = ', num2str(sqrt(Classifier.BRE_MSE_approx))]);
end
disp(' ');