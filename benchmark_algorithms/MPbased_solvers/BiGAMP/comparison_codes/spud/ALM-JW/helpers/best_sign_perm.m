function [P,sigma] = best_sign_perm(A,B)

% Given matrices A, B in \Re^{m \times n}, seek a permuation P of [n] such
% that the columns of A(P) are as similar to those of B as possible

addpath('perfect_matching');

[m,n] = size(A);

% compute (squared) distance matrix
D_plus = zeros(n,n);
D_minus = zeros(n,n);
D = zeros(n,n);
S = zeros(n,n);

for i = 1:n;
    for j = 1:n;
        D_plus(i,j) = (A(:,i) - B(:,j))' * (A(:,i)-B(:,j));
        D_minus(i,j) = (A(:,i) + B(:,j))' * (A(:,i) + B(:,j));
        
        if D_plus(i,j) < D_minus(i,j),
            D(i,j) = D_plus(i,j);
            S(i,j) = 1;
        else
            D(i,j) = D_minus(i,j);
            S(i,j) = -1;
        end
    end
end

[P,cost] = assignmentoptimal(D); 

sigma = zeros(size(P)); 

for i = 1:n, 
    sigma(i) = S(i,P(i));
end
