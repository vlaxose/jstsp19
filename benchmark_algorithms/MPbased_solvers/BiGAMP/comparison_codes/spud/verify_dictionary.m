function [relativeErrorA,relativeErrorX] = verify_dictionary_2(A_true,X_true,A_hat,X_hat)

% verify_dictionary_2
%
%   Compares factorizations (A_true,X_true) and (A_hat,X_hat)
%    in terms of the relative error, up to sign permutation ambiguity. That
%    is
%
%       minimize || A_true - A_hat Sigma Pi ||_F / || A_true ||_F
%       
%    where the minimization is over diagonal sign matrices Sigma and
%    permuation matrices Pi. 
%
%   Assumes the columns of both A_true and A_hat have unit L2 norm. 
%

% normalization
A_true=normalize_columns(A_true);
A_hat=normalize_columns(A_hat);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_true=normalize_columns(X_true')';
X_hat=normalize_columns(X_hat')';


% mod out sign-permutation ambiguity
[P,sigma] = best_sign_perm(A_true,A_hat);
A_hat = A_hat(:,P) * diag(sigma);
relativeErrorA = norm(A_hat-A_true,'fro')/norm(A_true,'fro');

[P,sigma] = best_sign_perm(X_true',X_hat');
X_hat = diag(sigma) * X_hat(P,:);
relativeErrorX = norm(X_hat-X_true,'fro')/norm(X_true,'fro');

function Y = normalize_columns(X)


stds=sqrt(diag(X'*X));
Y=X./(repmat(stds',[size(X,1),1]));
Y=Y./(repmat(sign(mean(Y,1)),[size(Y,1),1])+eps);