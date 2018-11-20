function result = pcgHelper(A,reg,truebits,x)

%This function computes
%(Amat(:,truebits)*Amat(:,truebits)' + reg*eye(M))*x;
%Where A is a LinTrans object implementing Amat
%truebits is a vector of logicals showing which elements to include

%This function is useful for computing GENIE estimators when A is not an
%explicit matrix

%First result, Amat(:,truebits)'
res1 = A.multTr(x);
res1(~truebits) = 0;

%Finish
result = A.mult(res1) + reg*x;

