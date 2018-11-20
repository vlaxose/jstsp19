% FastSepNMF - Fast and robust recursive algorithm for separable NMF
% 
% *** Description ***
% At each step of the algorithm, the column of M maximizing ||.||_2 is 
% extracted, and M is updated by projecting its columns onto the orthogonal 
% complement of the extracted column. 
%
% See N. Gillis and S.A. Vavasis, Fast and Robust Recursive Algorithms 
% for Separable Nonnegative Matrix Factorization, arXiv. 
% 
% [J,normM,U] = FastSepNMF(M,r,normalize) 
%
% ****** Input ******
% M = WH + N : a (normalized) noisy separable matrix, that is, W is full rank, 
%              H = [I,H']P where I is the identity matrix, H'>= 0 and its 
%              columns sum to at most one, P is a permutation matrix, and
%              N is sufficiently small. 
% r          : number of columns to be extracted. 
% normalize  : normalize=1 will scale the columns of M so that they sum to one,
%              hence matrix H will satisfy the assumption above for any
%              nonnegative separable matrix M. 
%              normalize=0 is the default value for which no scaling is
%              performed. For example, in hyperspectral imaging, this 
%              assumption is already satisfied and normalization is not
%              necessary. 
%
% ****** Output ******
% J        : index set of the extracted columns. 
% normM    : the l2-norm of the columns of the last residual matrix. 
% U        : normalized extracted columns of the residual. 
%
% --> normM and U can be used to continue the recursion later on without 
%     recomputing everything from scratch. 
%
% This implementation of the algorithm is based on the formula 
% ||(I-uu^T)v||^2 = ||v||^2 - (u^T v)^2. 

function [J,normM,U] = FastSepNMF(M,r,normalize) 

[m,n] = size(M); 

if nargin <= 2, normalize = 0; end
if normalize == 1
    % Normalization of the columns of M so that they sum to one
    D = spdiags((sum(M).^(-1))', 0, n, n); M = M*D; 
end

normM = sum(M.^2); 
nM = max(normM); 

i = 1; 
% Perform r recursion steps (unless the relative approximation error is 
% smaller than 10^-9)
while i <= r && max(normM)/nM > 1e-9 
    % Select the column of M with largest l2-norm
    [a,b] = max(normM); 
    % Norm of the columns of the input matrix M 
    if i == 1, normM1 = normM; end 
    % Check ties up to 1e-6 precision
    b = find((a-normM)/a <= 1e-6); 
    % In case of a tie, select column with largest norm of the input matrix M 
    if length(b) > 1, [c,d] = max(normM1(b)); b = b(d); end
    % Update the index set, and extracted column
    J(i) = b; U(:,i) = M(:,b); 
    
    % Compute (I-u_{i-1}u_{i-1}^T)...(I-u_1u_1^T) U(:,i), that is, 
    % R^(i)(:,J(i)), where R^(i) is the ith residual (with R^(1) = M).
    for j = 1 : i-1
        U(:,i) = U(:,i) - U(:,j)*(U(:,j)'*U(:,i));
    end
    % Normalize U(:,i)
    U(:,i) = U(:,i)/norm(U(:,i)); 
    
    % Compute v = u_i^T(I-u_{i-1}u_{i-1}^T)...(I-u_1u_1^T)
    v = U(:,i); 
    for j = i-1 : -1 : 1
        v = v - (v'*U(:,j))*U(:,j); 
    end
    
    % Update the norm of the columns of M after orhogonal projection using
    % the formula ||r^(i)_k||^2 = ||r^(i-1)_k||^2 - ( v^T m_k )^2 for all k. 
    normM = normM - (v'*M).^2; 
    
    i = i + 1; 
end