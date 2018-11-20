function [X_hat, numiter, X_path] = matrixALPSII(y, A, At, m, n, k, params, X)
% =========================================================================
%                Matrix ALPS II algorithm - Beta Version
% =========================================================================
% Matrix Algebraic Pursuit (ALPS) algorithm with memory acceleration. 
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
%    tau,...                Momentum step size selection. Default value:
%                           tau = 0 for optimum step size selection. 
%    tol,...                Early stopping tolerance. Default value: tol =
%                           1-e5
%    ALPSiters,...          Maximum number of algorithm iterations. Default
%                           value: 300. 
%    cg_maxiter,...         Maximum iterations for Conjugate-Gradients method.
%    cg_tol,...             Tolerance variable for Conjugate-Gradients method.
%    xpath,...              Set history log to null.
%    svdApprox,...          Set to nonzero value in (0,1) for SVD approximation
% =========================================================================
% 
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
% This work was suppo_cur - AX_prevrted in part by the European Commission under Grant 
% MIRG-268398 and DARPA KeCoM program #11-DARPA-1055. VC also would like 
% to acknowledge Rice University for his Faculty Fellowship.
% =========================================================================


%% Initialize to zero matrix
if (params.xpath == 1)
    X_path = zeros(1, params.ALPSiters);
end;

X_cur = zeros(m, n);
X_prev = zeros(m, n);
Q_cur = zeros(m, n);
AX_cur = zeros(length(y), 1);
Ucur = [];
UXcur = [];
options.tol = 10^-3;

I = eye(m,m);

i = 1;
%% Matrix ALPS II
while (i <= params.ALPSiters)
    if (params.xpath == 1)
        X_path(1,i) = norm(X_cur - X,'fro')/norm(X, 'fro');
    end;    

    %% Compute the residual
    if (i == 1)
        res = y;
    else                     
        res = y - (AX_cur + params.tau*(AX_cur - AX_prev));
    end;
    
    %% Compute the gradient
    grad = At(res);   
        
    %% Active subspace expansion step: Si (D := Pk(P_{Xi}^\bot grad))
    if (i == 1)
        [Uout, dump1, dump2] = lansvd(grad, k, 'L',options);
    else
        [Uout, dump1, dump2] = lansvd(ortho_UQ_i*grad, k, 'L',options); 
    end;
    
    USi = [Ucur Uout];
    
    %% Error norm reduction via gradient descent
    proj_grad = USi*(USi'*grad);
    % Step size selection
    mu = norm( proj_grad,'fro')^2/norm( A(proj_grad),2)^2;
    
    Vi = Q_cur + (mu)*proj_grad;
    
    %% Best rank-k subspace selection
    [UWi, SWi, VWi] = lansvd(Vi, k, 'L');  
    Wi = UWi*(SWi*VWi');
        
    %% Debias via gradient descent
    res = y - A(Wi);
    grad = At(res);
    
    proj_grad = UWi*(UWi'*grad);
    % Step size selection
    xi = norm( proj_grad,'fro')^2/norm( A(proj_grad),2)^2;
    
    X_cur = Wi + (xi)*proj_grad;
    UXprev = UXcur;
    UXcur = UWi;
    
    % Test stopping criterion
    if (i > 1) && (norm(X_cur - X_prev, 'fro') < params.tol*norm(X_cur, 'fro'))
        break;
    end;
    i = i + 1;  

    AX_prev = AX_cur;
    AX_cur = A(X_cur);
    AX = AX_cur - AX_prev;
    params.tau = ((y - AX_cur)'*(AX))/norm(AX, 2)^2;
    Q_cur = X_cur + params.tau*(X_cur - X_prev);
    X_prev = X_cur;
    
    Ucur = [UXcur UXprev];
    ortho_UQ_i = I - Ucur*Ucur';
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