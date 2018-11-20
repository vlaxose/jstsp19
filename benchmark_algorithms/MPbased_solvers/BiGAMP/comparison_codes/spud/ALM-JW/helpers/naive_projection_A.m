function [Ap,Xp] = naive_projection_A(A,X)

Ap = normalize_columns(A);
d  = sqrt(sum(A.*A,1));
Xp = diag(d) * X;