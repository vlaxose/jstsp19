function [ x_hat, indexSet, v, targetMatrix ] = OMP( A, v, m )
%Orthogonal Matching Pursuit using target sparisity factor(m)
% this was made for simulation and by no mean an effiecent implementation
% A - Sensing Matrix
% v - data vector
% m - sparsity level of x

d = length(v);
[measures, size_d]=size(A);
r = v;
targetMatrix=[];
indexSet = cell(1,m);
for t=1:m
    [maxInnerProduct, indexSet{t}] = max(abs(A'*r));
    targetMatrix = [ targetMatrix, A(:,indexSet{t})];
    x = targetMatrix\v;
    a = targetMatrix*x;
    r = v - a;
end

% return the estimation vecotr
x_hat = zeros(size_d,1);
i=1;
for t=1:length(indexSet)
    x_hat(indexSet{t}) = x(i);
    i= i + 1;
end
