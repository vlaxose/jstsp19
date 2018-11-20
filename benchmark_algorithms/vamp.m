function x = vamp(y, A, sigma, L)
 
  B = [real(A) -imag(A) ; imag(A) real(A)];
  b = [real(y) ; imag(y)];
 
  nx = size(B,2);
  MM = size(B,1);
 
  maxit = 100; % max iterations for VAMP
  tol   = min(1e-3,max(1e-6,sigma)); % stopping tolerance for VAMP
  damp  = 0.85; % damping for VAMP
  learnNoisePrec = false; % automatically tune the noise variance?
 
  % Decide on MAP or MMSE GAMP
  map = 0;
  % Create a random Gaussian vector
  xmean0 = zeros(nx,1);
  xvar0  = ones(nx,1);
 
  wvar = sigma; % 10^(-0.1*snr)*mean(abs(xmean0).^2+xvar0)*ones(nz,1);
 
  % Create an input estimation class corresponding to a Gaussian vector
  beta  =  L/nx; % probability of a non-zero coef
  xvar1 = xvar0/beta; % prior variance of non-zero x coefs  
  EstimIn = SparseScaEstim(CAwgnEstimIn(0,xvar1),beta);
 
  % Create an output estimation class corresponding to the Gaussian noise.
  % Note that the observation vector is passed to the class constructor, not
  % the gampEst function.       
  EstimOut = CAwgnEstimOut(b, wvar, map);
 
  [U,s,V] = svd(B);
  s  = diag(s);
  d  = [s.^2;zeros(MM-length(s),1)]; % need length(d) = M
 
  % setup VAMP
  vampOpt = VampGlmOpt();
  vampOpt.nitMax = maxit;
  vampOpt.tol = tol;
  vampOpt.damp = damp;
  vampOpt.learnGam1 = learnNoisePrec; % = learnGam1;
  vampOpt.verbose = false;
  vampOpt.U = U;
  vampOpt.d = d;
  vampOpt.r1init = eps*1i; 
 
  % run V-GAMP
  [x1,vampEstFin] = VampGlmEst(EstimIn, EstimOut, B, vampOpt);
  vampNMSEdB_ = vampEstFin.err1;
  vampNMSEdB_debiased_ = vampEstFin.err2;
  vampNit = vampEstFin.nit;
 
  % [~,vampEstFin] = VampSlmEst(EstimIn,yq_r,XXcon_r,vampOpt);
  x  = vampEstFin.x1(1:end/2) + 1j*vampEstFin.x1(end/2+1:end);
 
end