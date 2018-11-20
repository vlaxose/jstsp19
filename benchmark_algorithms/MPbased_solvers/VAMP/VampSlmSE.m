function mse = GecSlmSE(estInAvg,d,N,wvar,maxit,gam1init,damp)

if nargin<5
  maxit = 20;
end
if nargin<6
  gam1init = eps;
end
if nargin<7
  damp = 1;
end

gamw = 1/wvar;
lam = gamw*d; % only contains nonzero entries (usually M of them)
lam_ = [lam;zeros(N-length(lam),1)]; % all N entries

% run state-evolution
gam1 = gam1init; 
eps1 = 1/gam1; 
mse = nan(1,maxit);
for i=1:maxit
  % linear stage
  oneMinusAlf = (1/N)*sum(lam./(lam+gam1));
  sig = (1/oneMinusAlf)*lam./(lam+gam1); % only nonzero entries
  sig_ = [sig;zeros(N-length(sig),1)]; % all N entries
  gam2nodamp = oneMinusAlf./(1-oneMinusAlf).*gam1;
  eps2 = eps1*mean((1-sig_).^2) + (1/N)*sum((sig.^2)./lam);
  if i>1
    gam2damp = damp*gam2nodamp + (1-damp)*gam2old;
  else
    gam2damp = gam2nodamp;
  end
  gam2old = gam2damp;
  
  % nonlinear stage
  [mse2nodamp,xvar2] = estInAvg.mse(1/gam2damp,eps2);
  if i>1
    mse2damp = sqrt(damp)*mse2nodamp + (1-sqrt(damp))*mse2old;
  else
    mse2damp = mse2nodamp;
  end
  mse2old = mse2damp;
  eta2 = 1/xvar2; 
  gam1 = eta2 - gam2damp; % assumed precision on r1
  eps1 = (mse2damp*eta2^2-eps2*gam2damp^2)/gam1^2; ; % true error on r1

  % record mse
  mse(i) = mse2nodamp;
end %i
