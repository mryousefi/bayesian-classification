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

function [Params, TruePrior, AssumedPrior, Sample, Classifiers, Posterior, Results] = run_BRE(varargin)

for i = 1:length(varargin)/2
    name = varargin{2*i-1};
    value = varargin{2*i};
    switch name
        case 'sample_size'
            sample_size = value;
        case 'test_size'
            test_size = value;
        case 'irep'
            internal_rep = value;
        case 'M'
            Params.M = value;
        case 'd'
            Params.d = value;
        case 'sampling'
            Params.train.method = value;
        case 'true_c'
            TruePrior.c.type = value;
        case 'true_c-info'
            TruePrior.c.info = value;
        case 'true_structure'
            TruePrior.dist.structure = value;
        case 'true_type'
            TruePrior.dist.type = value;
        case 'true_mean'
            TruePrior.dist.mean.value = value;
        case 'true_mean-info'
            TruePrior.dist.mean.info = value;
        case 'true_cov'
            TruePrior.dist.cov.value = value;
        case 'true_cov-info'
            TruePrior.dist.cov.info = value;
        case 'assm_c'
            AssumedPrior.c.type = value;
        case 'assm_c-info'
            AssumedPrior.c.info = value;
        case 'assm_structure'
            AssumedPrior.dist.structure = value;
        case 'assm_type'
            AssumedPrior.dist.type = value;
        case 'assm_mean'
            AssumedPrior.dist.mean.value = value;
        case 'assm_mean-info'
            AssumedPrior.dist.mean.info = value;
        case 'assm_cov'
            AssumedPrior.dist.cov.value = value;
        case 'assm_cov-info'
            AssumedPrior.dist.cov.info = value;
        case 'n-synt'
            Params.synthetic.nsynt = value;
        case 'lambda'
            lambda = value;
        case 'classifiers'
            Params.OBC.do = false;
            Params.LDA.do = false;
            Params.QDA.do = false;
            Params.LSVM.do = false;
            Params.RBFSVM.do = false;
            
            for j = 1:length(value)
                switch value{j}
                    case 'OBC'
                        Params.OBC.do = true;
                    case 'LDA'
                        Params.LDA.do = true;
                    case 'QDA'
                        Params.QDA.do = true;
                    case 'LSVM'
                        Params.LSVM.do = true;
                    case 'RBFSVM'
                        Params.RBFSVM.do = true;
                    otherwise
                        disp('Error: unknown classifier.')
                end
            end
        case 'error_estimators'
            Params.BRE_EXACT.do = false;
            Params.MSE_EXACT.do = false;
            Params.BRE_APPROX.do = false;
            Params.MSE_APPROX.do = false;
            Params.TRUE.do = false;
            Params.RESUB.do = false;
            Params.CV.do = false;
            Params.LOO.do = false;
            Params.BOOT.do = false;
            
            for j = 1:length(value)
                switch value{j}
                    case 'BRE_EXACT'
                        Params.BRE_EXACT.do = true;
                    case 'MSE_EXACT'
                        Params.MSE_EXACT.do = true;
                    case 'BRE_APPROX'
                        Params.BRE_APPROX.do = true;
                    case 'MSE_APPROX'
                        Params.MSE_APPROX.do = true;
                    case 'RESUB'
                        Params.RESUB.do = true;
                    case 'CV'
                        Params.CV.do = true;
                    case 'LOO'
                        Params.LOO.do = true;
                    case 'BOOT'
                        Params.BOOT.do = true;
                    case 'TRUE'
                        Params.TRUE.do = true;
                    otherwise
                        error('Error: unknown error estimator.')
                end
            end
        otherwise
            disp(['Error: unknown input arguments: ''', name, '''. Default values are used.']);
            disp(' ')
            
            disp('sample_size = 100');
            sample_size = 100;
            
            disp('test_size = 100');
            test_size = 1000;
            
            disp('internal_rep = 1');
            internal_rep = 1;
            
            disp('Params.M = 2');
            Params.M = 2;
            
            disp('Params.d = 1');
            Params.d = 1;
            
            disp('Params.train.n = 50');
            Params.train.n = 50;
            
            disp('Params.train.method = ''stratified''');
            Params.train.method = 'stratified';
            
            disp('TruePrior.c.type = ''known''');
            TruePrior.c.type = 'known';
            
            disp('TruePrior.dist.structure = ''fixed''');
            TruePrior.dist.structure = 'fixed';
            
            disp('TruePrior.dist.mean.value(1, :) = zeros(Params.d, 1)');
            disp('TruePrior.dist.mean.value(2, :) = zeros(Params.d, 1)');
            TruePrior.dist.mean.value(1, :) = zeros(Params.d, 1);
            TruePrior.dist.mean.value(2, :) = zeros(Params.d, 1);
            
            disp('TruePrior.dist.cov.value{1} = eye(Params.d)');
            disp('TruePrior.dist.cov.value{2} = eye(Params.d)');
            TruePrior.dist.cov.value{1} = eye(Params.d);
            TruePrior.dist.cov.value{2} = eye(Params.d);
            
            disp('AssumedPrior.c.type = ''known''');
            AssumedPrior.c.type = 'known';
            
            disp('AssumedPrior.dist.structure = ''known''');
            AssumedPrior.dist.structure = 'known';
            
            disp('AssumedPrior.dist.type = ''indep''');
            AssumedPrior.dist.type = 'indep';
            
            disp('AssumedPrior.dist.mean.value(1, :) = zeros(Params.d, 1)');
            disp('AssumedPrior.dist.mean.value(2, :) = zeros(Params.d, 1)');
            AssumedPrior.dist.mean.value(1, :) = zeros(Params.d, 1);
            AssumedPrior.dist.mean.value(2, :) = zeros(Params.d, 1);
            
            disp('AssumedPrior.dist.mean.info(1) = 1');
            disp('AssumedPrior.dist.mean.info(2) = 1');
            AssumedPrior.dist.mean.info(1) = 1;
            AssumedPrior.dist.mean.info(2) = 1;
            
            disp('AssumedPrior.dist.cov.value{1} = eye(Params.d)');
            disp('AssumedPrior.dist.cov.value{2} = eye(Params.d)');
            AssumedPrior.dist.cov.value{1} = eye(Params.d);
            AssumedPrior.dist.cov.value{2} = eye(Params.d);
            
            disp('AssumedPrior.dist.cov.info(1) = 1 + Params.d');
            disp('AssumedPrior.dist.cov.info(2) = 1 + Params.d');
            AssumedPrior.dist.cov.info(1) = 1 + Params.d;
            AssumedPrior.dist.cov.info(2) = 1 + Params.d;
            
            disp('Params.synthetic.nsynt = 100000');
            Params.synthetic.nsynt = 100000;
            
            disp('lambda = ones(Params.M) - eye(Params.M)');
            lambda = ones(Params.M) - eye(Params.M);
            
            disp('Params.OBC.do = false');
            Params.OBC.do = false;
            
            disp('Params.LDA.do = false');
            Params.LDA.do = false;
            
            disp('Params.QDA.do = false');
            Params.QDA.do = false;
            
            disp('Params.LSVM.do = false');
            Params.LSVM.do = false;
            
            disp('Params.RBFSVM.do = false');
            Params.RBFSVM.do = false;
            
            disp('Params.BRE_EXACT.do = false');
            Params.BRE_EXACT.do = false;
            
            disp('Params.MSE_EXACT.do = false');
            Params.MSE_EXACT.do = false;
            
            disp('Params.BRE_APPROX.do = false');
            Params.BRE_APPROX.do = false;
            
            disp('Params.MSE_APPROX.do = false');
            Params.MSE_APPROX.do = false;
            
            disp('Params.RESUB.do = false');
            Params.RESUB.do = false;
            
            disp('Params.CV.do = false');
            Params.CV.do = false;
            
            disp('Params.LOO.do = false');
            Params.LOO.do = false;
            
            disp('Params.BOOT.do = false');
            Params.BOOT.do = false;
            
            disp('Params.TRUE.do = false');
            Params.TRUE.do = false;
    end
end

Params.synthetic.n = Params.synthetic.nsynt*ones(Params.M, 1);
Params.test.n = test_size;

% Set hyperparameters
TruePrior = define_priors(Params, TruePrior);
AssumedPrior = define_priors(Params, AssumedPrior);

% Generate parameters
[c, mu, Sigma] = generate_param(Params, TruePrior);

Sample = [];
% Sample.c = c;
% Sample.mu = mu;
% Sample.Sigma = Sigma;

Results = initialize_results(Params, length(sample_size), internal_rep);
for nt = 1:length(sample_size)
    for irep = 1:internal_rep
%         tic
        Params.train.n = sample_size(nt);
        % Generate data
        [n, x, n_tst, x_tst] = generate_data(Params, c, mu, Sigma);
%         Sample.n{nt, irep} = n;
%         Sample.x{nt, irep} = x;
%         Sample.n_tst{nt, irep} = n_tst;
%         Sample.x_tst{nt, irep} = x_tst;
        
        % Update prior to posterior
%         Posterior{nt, irep} = update_prior(Params, AssumedPrior, x, n);
        Posterior = update_prior(Params, AssumedPrior, x, n);
        
        % Train classifiers
%         Classifiers{nt, irep} = classifier_design(Params, Posterior{nt, irep}, lambda, x, n);
        Classifiers = classifier_design(Params, Posterior, lambda, x, n);

        
        % Estimate risk
%         Classifiers{nt, irep} = risk_estimation(Params, Posterior{nt, irep}, Classifiers{nt, irep}, lambda, x, n, x_tst, n_tst);
        Classifiers = risk_estimation(Params, AssumedPrior, Posterior, Classifiers, lambda, x, n, x_tst, n_tst);

        Results = summarize_results(Params, Classifiers, Results, nt, irep);
        % Report results
        %         if Params.OBC.do
        %             report_summary_to_screen(Params, Posterior{nt, irep}, Classifiers{nt, irep}.OBC);
        %         end
        %
        %         if Params.LDA.do
        %             report_summary_to_screen(Params, Posterior{nt, irep}, Classifiers{nt, irep}.LDA);
        %         end
        %
        %         if Params.QDA.do
        %             report_summary_to_screen(Params, Posterior{nt, irep}, Classifiers{nt, irep}.QDA);
        %         end
        %
        %         if Params.LSVM.do
        %             report_summary_to_screen(Params, Posterior{nt, irep}, Classifiers{nt, irep}.LSVM);
        %         end
        %
        %         if Params.RBFSVM.do
        %             report_summary_to_screen(Params, Posterior{nt, irep}, Classifiers{nt, irep}.RBFSVM);
        %         end
%         toc
    end
end