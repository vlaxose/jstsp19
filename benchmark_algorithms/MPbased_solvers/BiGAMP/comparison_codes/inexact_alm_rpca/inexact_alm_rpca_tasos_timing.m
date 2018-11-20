function [A_hat,E_hat,iter,time_data,estHist] = inexact_alm_rpca_tasos_timing(D, lambda, tol, maxIter,trueRank,error_function)

% Oct 2009
% This matlab code implements the inexact augmented Lagrange multiplier 
% method for Robust PCA.
%
% D - m x n matrix of observations/data (required input)
%
% lambda - weight on sparse error term in the cost function
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
% 
% Initialize A,E,Y,u
% while ~converged 
%   minimize (inexactly, update A and E only once)
%     L(A,E,Y,u) = |A|_* + lambda * |E|_1 + <Y,D-A-E> + mu/2 * |D-A-E|_F^2;
%   Y = Y + \mu * (D - A - E);
%   \mu = \rho * \mu;
% end
%
% Minming Chen, October 2009. Questions? v-minmch@microsoft.com ; 
% Arvind Ganesh (abalasu2@illinois.edu)
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing

%addpath PROPACK;

[m n] = size(D);

if nargin < 2
    lambda = 1 / sqrt(m);
end

if nargin < 3
    tol = 1e-7;
elseif tol == -1
    tol = 1e-7;
end

if nargin < 4
    maxIter = 1000;
elseif maxIter == -1
    maxIter = 1000;
end

% initialize
Y = D;
norm_two = lansvd(Y, 1, 'L');
norm_inf = norm( Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;

A_hat = zeros( m, n);
E_hat = zeros( m, n);
mu = 1.25/norm_two; % this one can be tuned
mu_bar = mu * 1e7;
rho = 1.5;          % this one can be tuned
%d_norm = norm(D, 'fro');

iter = 0;
total_svd = 0;
converged = false;
%stopCriterion = 1;
if isempty(trueRank)
    sv = 10;
else
    sv = trueRank;
end


%Preallocate storage
estHist.errZ = zeros(maxIter,1);
time_data = zeros(maxIter,1);


%Init Zold
Zold = 0;


while ~converged       
    iter = iter + 1;
    
    %Start timing
    tstart = tic;
    
    
    temp_T = D - A_hat + (1/mu)*Y;
    E_hat = max(temp_T - lambda/mu, 0);
    E_hat = E_hat+min(temp_T + lambda/mu, 0);

    if choosvd(n, sv) == 1
        [U S V] = lansvd(D - E_hat + (1/mu)*Y, sv, 'L');
    else
        [U S V] = svd(D - E_hat + (1/mu)*Y, 'econ');
    end
    diagS = diag(S);
    svp = length(find(diagS > 1/mu));
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end
    if ~isempty(trueRank)
        svp = trueRank;
        sv = trueRank;
    end
    
    A_hat = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)';    

    total_svd = total_svd + 1;
    
    Z = D - A_hat - E_hat;
    
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
        
    %% stop Criterion    
    
    %Save time
    if iter > 1
        time_data(iter) = toc(tstart) + time_data(iter-1);
    else
        time_data(iter) = toc(tstart);
    end
    
    
    %Save result
    estHist.errZ(iter) = error_function(A_hat); 
    
    
    %stopCriterion = norm(Z, 'fro') / d_norm;
    stopCriterion = norm(A_hat - Zold,'fro') /...
        norm(A_hat,'fro');
    if stopCriterion < tol
        converged = true;
    end
    Zold = A_hat;
    
    
    
    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end


%Trim results
time_data = time_data(1:iter);
estHist.errZ = estHist.errZ(1:iter);

