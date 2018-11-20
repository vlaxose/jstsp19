function mse = VampGlmSE(estInAvg,estOutAvg,d,N,del,maxit,gam1xinit,gam1zinit)

if nargin<6
  maxit = 20;
end
if nargin<7
  gam1xinit = 1e-8;
end
if nargin<8
  gam1zinit = 1e-8;
end

% run state-evolution
gam1x = gam1xinit; 
gam1z = gam1zinit; 
mse = nan(1,maxit);
for i=1:maxit
  % nonlinear stage
  eta1x = 1/estInAvg.mse(1/gam1x);
  gam2x = eta1x - gam1x;
  [mse1z,zvar] = estOutAvg.mse(1/gam1z);
% eta1z = 1/mse1z;
  eta1z = 1/zvar;
  gam2z = eta1z - gam1z; % = gamw in AWGN case
  
  % linear stage (note length(d)=min(M,N))
  alf = (1/N)*sum(d./(d+gam2x/gam2z)) - eps;
 %eta2x = gam2x./(1-alf); % reported but not used
 %eta2z = del*gam2z./alf; % reported but not used
  gam1x = gam2x.*alf./(1-alf);
  gam1z = gam2z.*(del-alf)./alf;

  % record mse
  mse(i) = 1/eta1x;
end %i
