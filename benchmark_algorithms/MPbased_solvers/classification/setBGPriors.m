function [ variance, sparsity, targetNorm, c_est, v_est] = setBGPriors(A, y, varargin)
% setBGPriors
% it then attempts to best initialize the priors
% Inputs: A is the training data
%         y are the labels
% alternatively, A could be a function handle defining A'z, in which case
% there must be a third input, N. (useful if data normalization is
% performed through operators rather than explicitly)

if isa(A, 'function_handle')
    explicit = 0;
    n = varargin{1};
else
    explicit = 1;
end

if explicit
    
    [m, n] = size(A);
    D = numel(unique(y));
    
    % estimate class conditional cloud variance (this will automatically
    % remove the mean)
    v_est = zeros(D,1);
    for j = 1:D
        a = A(y==j,:);
        if size(a,1) == 1
            v_est(j) = var(a);
        else
            v_est(j) = mean(1/(size(a,1)-1) * (sum(a.^2,1) - 2 * sum(a,1).*mean(a) + size(a,1)*mean(a).^2) );
        end
    end
    
    % estimated cloud variance
    v_est = mean(v_est);
    
    % estimate means
    b = zeros(D,n);
    for j = 1:D
        b(j,:) = mean(A(y==j,:));
    end
    
    % calculate W
    W = b*b';

    % compute bias term
    bias = v_est/(m/D)*n;
    
    % subtract bias from the diagonal
    W(1:(D+1):(D^2)) = W(1:(D+1):(D^2)) - bias;
    
    meanDiag = mean(diag(W));
    meanOffDiag = mean(W(setdiff([1:D^2],[1:(D+1):(D^2)])));
    
    % estimated class mean length
    c_est = sqrt(abs(meanDiag - meanOffDiag));
  
    targetNorm = c_est/v_est;
  
    % % estimate BER via numerical integration  (optional)
    % npts = 1e3;
    % z = linspace(-6,6,npts) + c/sqrt(v);
    % dz = z(2) - z(1);
    %
    % % integrand definition
    % Pc=normpdf(z,c/sqrt(v),1).*(normcdf(z,0,1).^(D-1));
    %
    % % integration
    % BERest = 1 - trapz(Pc)*dz;
    % fprintf(1,'estimated ber is %4.3f\n',BERest)
    
    
    % initialize sparsity according to :
    % M >= 1/log2(D) * K * D * log2(N/K)
    sparsity = nan;
    for k = 1:n
        val = m - 1/log2(D) * k * D * log2(n/k);
        %   val = m - 1/log2(D) * k * 1 * log2(n/k);
        if val < 0
            sparsity = (k + 1)/n;
            break;
        end
    end
    if isnan(sparsity)
        sparsity = 1;
    end
    sparsity = min(sparsity, .5); % it is recommended to not use too large of a sparsity value, even if M is large
    variance = targetNorm^2 / (n * sparsity);
   
else
    
    m = numel(y);
    D = numel(unique(y));
    
    v_est = zeros(D,1);
    
    % estimate class conditional cloud variance and class means
    v_est = zeros(D,1);
    b = zeros(D,n);
    for j = 1:D
        se = speye(m);
        se = se(y==j,:);
        a = sparse((A(se'))');
        if size(a,1) == 1
            v_est(j) = var(a);
        else
            v_est(j) = mean(1/(size(a,1)-1) * (sum(a.^2,1) - 2 * sum(a,1).*mean(a) + size(a,1)*mean(a).^2) );
        end
        b(j,:) = mean(a);
    end
    
    % estimated cloud variance
    v_est = mean(v_est);
    
    % calculate W
    W = b*b';
    
    % compute bias term
    bias = v_est/(m/D)*n;
    
    % subtract bias from the diagonal
    W(1:(D+1):(D^2)) = W(1:(D+1):(D^2)) - bias;
    
    meanDiag = mean(diag(W));
    meanOffDiag = mean(W(setdiff([1:D^2],[1:(D+1):(D^2)])));
    
    % estimated class mean length
    c_est = sqrt(abs(meanDiag - meanOffDiag));
    
    targetNorm = c_est/v_est;
    
    % initialize sparsity according to :
    % m >= 1/log2(D) * K * D * log2(N/K)
    sparsity = nan;
    for k = 1:n
        val = m - 1/log2(D) * k * D * log2(n/k);
        %   val = m - 1/log2(D) * k * 1 * log2(n/k);
        if val < 0
            sparsity = (k + 1)/n;
            break;
        end
    end
    if isnan(sparsity)
        sparsity = 1;
    end
    sparsity = min(sparsity, .5); % it is recommended to not use too large of a sparsity value, even if M is large
    variance = targetNorm^2 / (n * sparsity);
    
end


end
