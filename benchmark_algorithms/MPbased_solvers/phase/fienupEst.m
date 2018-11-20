% sparse Fienup / Gerchberg-Saxton phase retrieval
%
% [xhat,numTry,xhatHist] = fienupEst(yabs,N,A,Ainv,K,opt)
%
% yabs : M-by-1 vector of output magnitudes  
%    N : length of input vector
%    A : handle to forward-operator function
% Ainv : handle to inverse-operator function
%    K : sparsity of input vector 
%  opt.maxTry : maximum number of re-tries
%  optFU.nresStopdB : residual stopping tolerance [suggest 10^(-(SNRdB+2)/10)]
%  opt.nit : max iterations per try
%  opt.tol : iteration stopping tolerance [suggest min(10^(-SNRdB/10),1e-4) ]
%  opt.xreal : true if the input vector is real
%
% xhat : estimated coefficients 
% numTry : number of re-tries
% xhatHist : history of xhat over iterations for best try
   
function [xhat_best,numTry,xhatHist_best] = fienupEst(yabs,N,A,Ainv,K,opt)

% setup
M = length(yabs);
if isempty(opt),
  maxTry = 200;         % maximum # re-tries
  nresStopdB = 10^(-(40+2)/10); % re-try stopping tolerance
  nit = 250;            % maximum # iterations
  tol = 1e-4;           % iteration stopping tolerance
  xreal = false;        % assume complex-valued signal
else
  maxTry = opt.maxTry;
  nresStopdB = opt.nresStopdB;
  nit = opt.nit;
  tol = opt.tol;
  xreal = opt.xreal;
end;
if nargout>2,
  xhatHist = nan(N,nit);
  keepHist = true;
else
  keepHist = false;
end;

% iterate
res_best = inf;
xhat_best = inf;
for t=1:maxTry,

  % random initialization
  if xreal
    xhat = randn(N,1);
  else
    xhat = randn(N,2)*[1;1i];
  end
  yhat = yabs.*sign(A(xhat));

  % iterate
  for iter=1:nit,

    % backward projection
    xhat_old = xhat;
    xhat = Ainv(yhat); 

    % denoise
    if xreal, xhat = real(xhat); end;
    if K<N % sparsify
      [~,indx] = sort(abs(xhat),1,'descend');
      xbig = xhat(indx(1:K));
      xhat = zeros(N,1);
      xhat(indx(1:K)) = xbig;
    end
    if keepHist, xhatHist(:,iter) = xhat; end;

    % forward projection
    zhat = A(xhat);
    yhat = yabs.*sign(zhat);

    % check stopping tolerance
    if norm(xhat_old-xhat)/norm(xhat)<tol, break; end;

  end;%iter
  if keepHist, xhatHist = xhatHist(:,1:iter); end;

  % check residual
  res = (norm(yabs-abs(zhat))/norm(yabs))^2; 
  if res<res_best
    xhat_best = xhat;
    res_best = res;
    if keepHist, xhatHist_best = xhatHist; end;
  end
  if (res_best<nresStopdB)
    break;
  end

end;%t
numTry = t;
