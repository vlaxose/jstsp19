function [X_hat, numiter, X_path] = matrixALPSI(y, A, At, m, n, k, params, X)
% =========================================================================
%                Matrix ALPS I algorithm - Beta Version
% =========================================================================
% Matrix Algebraic Pursuit (ALPS) algorithm with memoryless acceleration. 
% 
% Detailed discussion on the algorithm can be found in 
% [1] "Matrix Recipes for Hard Thresholding Methods", written
% by Anastasios Kyrillidis and Volkan Cevher, Technical Report, 2011.
% =========================================================================
% INPUT ARGUMENTS:
% y                         p x 1 undersampled measurement vector.
% A, At                     Linear operator and its adjoint:
%                           A: m x n |-> p x 1 mapping.
%                           At: p x 1 |-> m x n mappting
% k                         Rank prior information of X* or desired
%                           rank of computed solution.
% m, n                      Dimensions
% params                    Structure of parameters. These are:
%
%    tol,...                Early stopping tolerance. Default value: tol =
%                           1-e5
%    ALPSiters,...          Maximum number of algorithm iterations. Default
%                           value: 300. 
%    cg_maxiter,...         Maximum iterations for Conjugate-Gradients method.
%    cg_tol,...             Tolerance variable for Conjugate-Gradients method.
%    xpath,...              Set history log to null.
%    svdApprox,...          Set to nonzero value in (0,1) for SVD approximation
% =========================================================================
% OUTPUT ARGUMENTS:
% X_hat                     m x n recovered rank-k matrix.
% numiter                   Number of iterations executed.
% X_path                    Keeps a series of computed m x n low-rank matrices 
%                           until the end of the iterative process. In
%                           case this information is not needed, please set
%                           params.xpath = 0
% =========================================================================
% 09/12/2011, by Anastasios Kyrillidis. anastasios.kyrillidis@epfl.ch, EPFL.
% =========================================================================
% cgsolve.m is written by Justin Romberg, Caltech, Oct. 2005.
%                         Email: jrom@acm.caltech.edu
% =========================================================================
% This work was supported in part by the European Commission under Grant 
% MIRG-268398 and DARPA KeCoM program #11-DARPA-1055. VC also would like 
% to acknowledge Rice University for his Faculty Fellowship.
% =========================================================================


%% Initialize to zero matrix
if (params.xpath == 1)
    X_path = zeros(1, params.ALPSiters);
end;

X_cur = zeros(m, n);
Ucur = [];

% params.svdMode = 'svds';
I = eye(m, m);
i = 1;
%% Matrix ALPS II
while (i <= params.ALPSiters)
    if (params.xpath == 1)
        X_path(1,i) = norm(X_cur - X,'fro')/norm(X, 'fro');
    end;
    X_prev = X_cur;

    %% Compute the residual
    if (i == 1)
        res = y;
    else                 
        res = y - A(X_cur);
    end;
    
    %% Compute the gradient
    grad = At(res);   
        
    %% Active subspace expansion step: Si (D := Pk(P_{Xi}^\bot grad))
    if (i == 1)
        [Uout] = active_subspace_exp(grad, [], k, params.svdMode, params.svdApprox);
    else
        [Uout] = active_subspace_exp(grad, ortho_UX_i, k, params.svdMode, params.svdApprox);
    end;
    
    USi = [Ucur Uout];
    
    %% Error norm reduction via gradient descent
    proj_grad = USi*USi'*grad;
    % Step size selection
    mu = norm( proj_grad,'fro')^2/norm( A(proj_grad),2)^2;
    
    Vi = X_cur + (mu)*proj_grad;
    
    %% Best rank-k subspace selection
    [Wi, Ucur, ~] = lowrank_subspace_selection(Vi, k, params.svdMode, params.svdApprox);
        
    %% Debias via gradient descent
    res = y - A(Wi);
    grad = At(res);
    
    proj_grad = Ucur*Ucur'*grad;
    % Step size selection
    xi = norm( proj_grad,'fro')^2/norm( A(proj_grad),2)^2;
    
    X_cur = Wi + (xi)*proj_grad;
    
    % Test stopping criterion
    if (i > 1) && (norm(X_cur - X_prev, 'fro') < params.tol*norm(X_cur, 'fro'))
        break;
    end;
    i = i + 1;  

    ortho_UX_i = (I - Ucur*Ucur');
    
end;

X_hat = X_cur;
numiter = i;

if (params.xpath == 1)
    if (i > params.ALPSiters)
        X_path = X_path(1,1:numiter-1);
        numiter = numiter - 1;
    else
        X_path = X_path(1,1:numiter);
    end;
else X_path = [];
end;