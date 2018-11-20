%Simple program to test LinTransDiag
clear all
clc

%Build 3 matrices
C = randn(3,17);
D = randn(19,7);
E = randn(23,23);

%Build diag
A1 = MatrixLinTrans(blkdiag(C,D,E));
A2 = LinTransDiag({MatrixLinTrans(C),MatrixLinTrans(D),MatrixLinTrans(E)});

%Get size
[M,N] = A1.size;

%Try a mult
x = randn(N,5);
y1 = A1*x;
y2 = A2*x;
disp(['Mult error was ' num2str(norm(y1 - y2,'fro'))])

%Try a multTr
y = randn(M,5);
x1 = A1'*y;
x2 = A2'*y;
disp(['Mult Transpose error was ' num2str(norm(x1 - x2,'fro'))])

%Check consistency
A1.consCheck
A2.consCheck





