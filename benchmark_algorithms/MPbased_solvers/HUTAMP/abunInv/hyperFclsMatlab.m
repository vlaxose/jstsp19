function [ W ] = hyperFclsMatlab(Y, S)
%HYPERFCLS Performs fully constrained least squares on pixels of Y using the
%Matlab optimzation toolbox method.
%   Y is a hyperspectral data, S is a matrix of endmembers.
% Performs fully constrained least squares of each pixel in Y using the
% endmember signatures of S.  The optimization procedure used is from the Matlab
% optimization toolbox.
% Fully constrained least squares is least squares with the abundance sum-to-one
% constraint (ASC) and the abundance nonnegative constraint (ANC).
%

options = optimset('Diagnostics','off','Display','off','LargeScale','off');

[M,T] = size(Y);
N = size(S,2);

W = zeros(N,T);
for t = 1:T       
     W(:,t) = lsqlin(S, Y(:,t), [], [], ones(1,N), 1, zeros(N,1),[], [],options);
end

return