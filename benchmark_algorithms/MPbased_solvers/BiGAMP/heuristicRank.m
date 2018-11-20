function rankGuess = heuristicRank(X,tau)

%Looks for a drop in the singular values of X that is a factor of tau
%greater than the average drop. If such a value is found, the location of
%this drop is taken as an estimate for the rank of the matrix X

%Compute singular values of X
d = svd(X) + eps;

%Get assumed rank
N = length(d);

%Compute ratios
dtilde = d(1:(end-1)) ./ d(2:end);

%Compute ratio
[dtilde_max,p] = max(dtilde);
ratval = (N-2)*dtilde_max / (sum(dtilde) - dtilde_max);

%Check ratio
if ratval > tau
    rankGuess = p;
else
    rankGuess = [];
end



