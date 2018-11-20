function [ error_rate] = testErrorRate(X,mu,Va)

% analytic test error rate using orthogonal class mean model and assuming
% equal priors

% find the number of classes
D = size(X,2);

Pd = nan(D,1);

% Compute mean of means
% r = mean(mu,2);

% compute SigmaZ
SigmaZ = Va*(X')*X;

% compute the conditional probability of error given Y=y
for y = 1:D
    
    % compute means of z
%     Mu_z = X'*(mu(:,y) - r);
    Mu_z = X'*(mu(:,y));
    
    % compute transformation matrix c
    c = eye(D);
    c(:,y) = -1;
    c(y,:) = [];
    
    % transform means
    Mu_c = c * Mu_z;
    
    % transform covariance matrix
    SigmaC = c * SigmaZ * c';

    % use MVNCDF to compute error probability
    try
        Pd(y) = mvncdf(zeros(1,D-1),Mu_c',SigmaC);
    catch
        Pd(y) = nan;
    end
    
end

error_rate = 1 - mean(Pd);

end