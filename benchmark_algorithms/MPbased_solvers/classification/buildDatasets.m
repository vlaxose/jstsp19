function [ y_train, A_train, mu, v] = buildDatasets(D,M,N,K,BER)
% function [ y_train, A_train, y_test, A_test, mu] = mcData(D,M,N,K,BER)
%
% Generate multiclass data. 
% Assumes D>=K
% Class means are orthogonal and K-sparse
% A_train is normalized to have unit variance (as an entire matrix, not
% per-column). This improves numerical robustness of GAMP and leads to
% more consistent performance. 


if K < D
    error('multiclass dataset model assumes K>=D')
end

% generate K sparse, orthonormal class means
mu = [eye(D),zeros(D,K-D)];
[U,~,~]=svd(randn(K));
mu=(U*(mu)')';
mu=[mu zeros(D, N - K)];

% duplicate to have ~M/D copies of mu
MU = repmat(mu,ceil(M/D),1);
MU = MU(1:M,:);

mu = mu';

% determine appropriate variance based off Bayes Error Rate, then add noise
v = BayesError2variance(BER,D);
A_train = MU + sqrt(v)*randn(M,N);

% make entire matrix unit variance. this improves numerical robustness, and
% does not fundamentally change the data model. 
c = std(A_train(:));
A_train = A_train/c;
mu = mu/c;
v = v/c^2;

% generate training labels
y_train = repmat(1:D,1,ceil(M/D));
y_train = y_train(1:M)';

end

