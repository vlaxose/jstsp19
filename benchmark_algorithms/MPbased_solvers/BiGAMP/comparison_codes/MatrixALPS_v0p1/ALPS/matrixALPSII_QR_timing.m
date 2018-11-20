function [X_hat, numiter, X_path, estHist,time_data] = matrixALPSII_QR_timing(y, A, At, m, n, k, params, X,error_function)
% =========================================================================
%                Matrix ALPS II algorithm with QR - Beta Version
% =========================================================================
% Matrix Algebraic Pursuit (ALPS) algorithm with memory acceleration and
% random projection via QR decomposition.
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

I = eye(n,n);

%Preallocate storage
estHist.errZ = zeros(params.ALPSiters,1);
time_data = zeros(params.ALPSiters,1);


i = 1;
%% Matrix ALPS II
while (i <= params.ALPSiters)
    if (params.xpath == 1)
        %X_path(1,i) = norm(X_cur - X,'fro')/norm(X, 'fro');
    end;
    
    %Start timing
    tstart = tic;
    
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
        Y2 = randn(n, k);
        for j = 1:params.power
            Y1 = grad*Y2;
            Y2 = grad'*Y1;
        end;
        [Uout, ~] = qr(Y2, 0);
    else
        orthograd = grad*ortho_UQ_i;
        Y2 = randn(n, k);
        for j = 1:params.power
            Y1 = orthograd*Y2;
            Y2 = orthograd'*Y1;
        end;
        [Uout, ~] = qr(Y2, 0);
    end;
    
    USi = [Ucur Uout];
    
    %% Error norm reduction via gradient descent
    proj_grad = (grad*USi)*USi';
    % Step size selection
    mu = norm( proj_grad,'fro')^2/norm( A(proj_grad),2)^2;
    
    Vi = Q_cur + (mu)*proj_grad;
    
    %% Best rank-k subspace selection
    Y2 = randn(n, k);
    for j = 1:params.power
        Y1 = Vi*Y2;
        Y2 = Vi'*Y1;
    end;
    [UWi, ~] = qr(Y2, 0);
    
    Wi = (Vi*UWi)*UWi';
    
    %% Debias via gradient descent
    res = y - A(Wi);
    grad = At(res);
    
    proj_grad = (grad*UWi)*UWi';
    % Step size selection
    xi = norm( proj_grad,'fro')^2/norm( A(proj_grad),2)^2;
    
    X_cur = Wi + (xi)*proj_grad;
    UXprev = UXcur;
    UXcur = UWi;
    
    
    
    AX_prev = AX_cur;
    AX_cur = A(X_cur);
    AX = AX_cur - AX_prev;
    params.tau = ((y - AX_cur)'*(AX))/norm(AX, 2)^2;
    Q_cur = X_cur + params.tau*(X_cur - X_prev);
    
    
    Ucur = [UXcur UXprev];
    ortho_UQ_i = I - Ucur*Ucur';
    
    %Save time
    if i > 1
        time_data(i) = toc(tstart) + time_data(i-1);
    else
        time_data(i) = toc(tstart);
    end
    
    
    %Save result
    estHist.errZ(i) = error_function(X_cur');
    
    % Test stopping criterion
    if (i > 1) && (norm(X_cur - X_prev, 'fro') < params.tol*norm(X_cur, 'fro'))
        break;
    end;
    i = i + 1;
    
    X_prev = X_cur;
end;

X_hat = X_cur;
numiter = i;

%Trim results
if numiter < length(time_data)
    time_data = time_data(1:numiter);
    estHist.errZ = estHist.errZ(1:numiter);
end

if (params.xpath == 1)
    if (i > params.ALPSiters)
        X_path = X_path(1,1:numiter-1);
        numiter = numiter - 1;
    else
        X_path = X_path(1,1:numiter);
    end;
else X_path = [];
end;