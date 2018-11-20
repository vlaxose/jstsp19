function A = randCGM(row, col, lambda, mean, var)
%randCGM is a function that creates a matrix of i.i.d random variables
%belonging to a complex gaussian mixture.  The gaussian mixture contains L
%gaussian components, each with a weight, mean, and variance.
%
%Inputs
%
%   row, col    dimensions of output matrix desired
%   lambda      vector containing the weights of each gaussian component.
%   mean        vector containing the means of each Ggaussian component
%   var         vector containing the means of each gaussian component
%
%NOTE: lambda,mean,var must all be the same length.
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 4/4/12
% Change summary: 
%   v 1.0 (JV)- First release
%
% Version 2.0
%%
if nargin == 0
    %Default case return one element of N(0,1)
    A = randn(1,1);

else
    
    L = length(mean);
    dummy = [0 cumsum(lambda)];
    dummy2 = rand(row,col);

    dummy3 = zeros(row,col,L);
    for i = 1:L
        dummy3(:,:,i) = (dummy2>= dummy(i) & dummy2<dummy(i+1)).*(mean(i)+sqrt(var(i)/2).*(randn(row,col)+1i*randn(row,col)));
    end;

    A = sum(dummy3,3);
end
return