 function [X_hat, iter, X_path] = inexact_alm_mc(D, tol, maxIter, x_path, k, X)

% Oct 2009
% This matlab code implements the inexact augmented Lagrange multiplier 
% method for Matrix Completion.
%
% D - m x n matrix of observations/data (required input)
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
%
% Model: 
%     min |A|_*
%     subj A + E = D, ProjectionOnOmega(E) = 0
%
% Algorithm:
%
% Initialize A,E,Y,u
% while ~converged 
%   minimize (inexactly, update A and E only once)
%     L(A,E,Y,u) = |A|_* + <Y,D-A-E> + mu/2 * |D-A-E|_F^2;
%   Y = Y + \mu * (D - A - E);
%   \mu = \rho * \mu;
% end
%
% Minming Chen, October 2009. Questions? v-minmch@microsoft.com ; 
% Arvind Ganesh (abalasu2@illinois.edu)
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing

clear global;
global A Sparse_Z;

% addpath PROPACK;

[m n] = size(D);

if nargin < 2
    tol = 1e-4;
elseif tol == -1
    tol = 1e-4;
end

if nargin < 3
    maxIter = 1000;
elseif maxIter == -1
    maxIter = 1000;
end

% read sparse matrix D
[m n] = size(D);
[I J V] = find(D);
p = length(I);
col = [0; find(diff(J)); p];
Sparse_Z = D;
clear D;

% initialize
Y = zeros(p, 1);
Z = zeros(p, 1);
A.U = zeros(m, 5);
A.V = zeros(n, 5);
d_norm = norm(V, 'fro');
mu = 0.3/(lansvd('Axz','Atxz',m,n,1,'L'));
% mu = 0.3/(lansvd(A.U*A.V',1,'L'));
rho_s = p / (m * n);
rho = 1.1 + 2.5 * rho_s;

sv = 2;
svp = sv;

% Iteration 
iter = 0;
converged = false;
stopCriterion = 1;
X_cur = [];
X_prev = [];

if (x_path == 1)
    X_path = zeros(1, maxIter);
end;

while ~converged         
    %% alternative projection 
    
    iter = iter + 1;
    if iter == 1
        Z = V;
    else
        Z = Z + 1/mu * Y;
        Sparse_Z = spconvert([I,J,Z; m,n,0]);
    end
    if stopCriterion > 10 * tol
        options.tol = 10*tol;
    else
        options.tol = min(0.1*tol, 0.01/mu);
    end
%     [A.U,S,A.V] = lansvd('Axz','Atxz',m,n,sv,'L',options);      
%     [A.U,S,A.V] = lansvd('Axz', 'Atxz', m, n, k, 'L', options);      
    [A.U,S,A.V] = lansvd(A.U*A.V' + Sparse_Z, sv, 'L', options);      
        
%     %% predict the rank of A.
    diagS = diag(S);
    diagS = diagS(1:sv);
    svn = length(find(diagS > 1/mu));
    svp = svn;
%     
    ratio = diagS(1:end-1)./diagS(2:end);
    [max_ratio, max_idx] = max(ratio);
    if max_ratio > 2
        svp = min(svn, max_idx);
        svp = k;
    end
    if svp < sv %|| iter < 10
        sv = min(svp + 1, n);
    else
        sv = min(svp + 10, n);
    end
    sv = k;
       
%     svp = k;
    %% update A Y Z mu
    sqrtds = sqrt(diagS(1:svp) - 1/mu);
    A.U = A.U(:, 1:svp) * diag(sqrtds);
    A.V = A.V(:, 1:svp) * diag(sqrtds);   
    
    X_prev = X_cur;
    X_cur = A.U*A.V';
    
    if (x_path == 1)
        X_path(1, iter) = norm(X_cur - X, 'fro')/norm(X, 'fro');
    end;   
    
    Z = UVtOmega(A.U,A.V,I,J,col);
    Z = V - Z;
    Y = Y + mu*Z;
        
    %% stop Criterion 
    % Test stopping criterion
    if (iter > 1) && (norm(X_cur - X_prev, 'fro') < tol*norm(X_cur, 'fro'))
        converged = true;
    end;
    
    stopCriterion = norm(Z, 'fro') / d_norm;
%     if stopCriterion < tol
%         converged = true;
%     end    
%     
%     %% display    
%     if mod( iter, 10) == 0
%         disp(['#svd ' num2str(iter) ' r(A) ' num2str(svp)...
%             ' svn ' num2str(svn) ' max_idx ' num2str(max_idx) ' sv ' num2str(sv) ...
%             ' stopCriterion ' num2str(stopCriterion) ' mu ' num2str(mu)]);
%     end    
    
    %% Maximum iterations reached
    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
    
    %% update mu
    mu = mu*rho;
    
end

if (x_path == 1)
    X_path = X_path(1,1:iter);
else X_path = [];
end;

% X_path = [];
X_hat = A.U*A.V';
