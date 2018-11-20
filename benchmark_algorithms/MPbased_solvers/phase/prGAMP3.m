% prGAMP3 Phase-Retrieval Generalized Approximate Message Passing version 3.0
%
%   prGAMP3 uses Rangan's Generalized AMP algorithm to perform phase 
%   retrieval, i.e., to estimate the length-N signal vector x given 
%   knowledge of A and abs(y), in the additive linear model "y=A*x+w".
%   Here, the noise is assumed to be independent zero-mean Gaussian 
%   and the signal is assumed to be Bernoulli-Gaussian.
%
%   This code is a part of the GAMPmatlab package.
%
% SYNTAX: 
%
%   [xhat,out,estHist] = prGAMP3(abs_y, A, opt_pr)
%
% DEMO MODE: 
%
%   run without any inputs to see a demo.
%
% STANDARD OUTPUT:
%
%   xhat  : estimated length-N signal such that abs_y ~= abs(A*xhat)
%
% OPTIONAL OUTPUTS:
%
%   out.xvar0: estimated signal variance
%   out.wvar: estimated noise variance 
%   out.SNRdB_est: estimated SNR in dB 
%   out.numTry: number of re-initialization attempts
%   out.support_size = estimate support size 
%
%   estHist : GAMPmatlab structure detailing iteration history 
%
% MANDITORY INPUTS:
%
%   abs_y : observed length-M vector of magnitudes 
%   A     : transform matrix or GAMPmatlab LinTrans object 
%
% IMPORTANT OPTIONAL INPUTS:
%
%   opt_pr.xreal : 1 if signal vector is real-valued, or 0 otherwise (default=0)
%   opt_pr.xnonneg : 1 if signal vector is non-negative, or 0 otherwise (default=0)
%   opt_pr.sparseRat : signal sparsity rate in (0,1] (default=1)
%   opt_pr.SNRdBinit : SNR (in dB) to initialize noise variance (default=10)
%   opt_pr.SNRdB : SNR (in dB) used for terminating re-starts (default=60)
%   opt_pr.maxTry : max number of re-starts from random initializations (default=10)
%   opt_pr.verbose : print the residual? (default=1)
%
% LESS IMPORTANT OPTIONAL INPUTS:
%
%   opt_pr.nresStopdB : residual level at which to declare success (default=-(SNRdB+2))
%   opt_pr.sparseRatTry : max sparsity ratio tolerated for success (default=inf)
%   opt_pr.adaptStep : toggles use of GAMP adaptive stepsize (default=1)
%   opt_pr.nitEM : max number of EM iterations to tune SNRdB (default=20)
%   opt_pr.minTry : min number of tries from random initializations (default=1)
%   opt_pr.maxTol : maximum value of SNR-adapted stopping tolerance (default=1e-4)
%   opt_pr.xvar0 : length-N vector of non-zero-signal variances
%                  ...(default is to compute from abs_y, ||A||_F, sparseRat, SNRdB)
%   opt_pr.xmean0 : length-N vector of non-zero-signal means (default = 0)
%   opt_pr.xhat0 : length-N signal initialization on first try (default is random)
%   opt_pr.plot : plot the history of the residual, cost, and stepsize? (default=0)

function [xhat,out,estHist_] = prGAMP3(abs_y,A,opt_pr,opt_gamp_user)
%nargin=0; nargout=1;

 if nargout==3
   computeHist = 1;
 elseif nargout<3
   computeHist = 0;
 elseif nargout>3
   error('Too many output arguments')
 end

 if nargin==0,
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % begin demonstration mode %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
  % handle random seed
  if verLessThan('matlab','7.14')
    defaultStream = RandStream.getDefaultStream;
  else
    defaultStream = RandStream.getGlobalStream;
  end;
  if 0 % new RANDOM trial
    savedState = defaultStream.State;
    save random_state.mat savedState;
  else % repeat last trial
    load random_state.mat
  end
  defaultStream.State = savedState;

  % load demonstration scenarios 
  DEMO = 1;
  switch DEMO
      case 1 % iid matrix 0.25x oversampled, complex sparse signal, high SNR
          A_type = 'iid'; % one of 'iid', 'rdft', 'mdft', 'odft', 'o2dft', 'm2dft'
                        % ...see definitions in generatePRLinTrans.m
          SNRdB_true = 80;      % true SNR in dB. 
          nx = 512;     % # signal coefficients
          nz = nx/4;    % # phaseless measurements 
          nk = 8;       % signal sparsity [8]
          xreal_true = 0;       % complex-valued signal
          xnonneg_true = 0;     % signal is not non-negative (only matters when xreal_true=1)
          run_fu = 0;   % don't run fienup: only works for orthogonal A
          
      case 2 % iid matrix 0.25x oversampled, complex sparse signal, low SNR
          A_type = 'iid'; % one of 'iid', 'rdft', 'mdft', 'odft', 'o2dft', 'm2dft'
                        % ...see definitions in generatePRLinTrans.m
          SNRdB_true = 10;       % true SNR in dB. 
          nx = 512;     % # signal coefficients
          nz = nx/4;    % # phaseless measurements 
          nk = 4;       % signal sparsity [4]
          xreal_true = 0;       % complex-valued signal
          xnonneg_true = 0;     % signal is not non-negative (only matters when xreal_true=1)
          run_fu = 0;   % don't run fienup: only works for orthogonal A
          
      case 3 % iid matrix 4x oversampled, complex non-sparse signal, high SNR
          A_type = 'iid'; % one of 'iid', 'rdft', 'mdft', 'odft', 'o2dft', 'm2dft'
                        % ...see definitions in generatePRLinTrans.m
          SNRdB_true = 80;      % true SNR in dB. 
          nx = 512;     % # signal coefficients
          nz = 4*nx;    % # phaseless measurements 
          nk = nx;      % signal sparsity
          xreal_true = 0;       % complex-valued signal
          xnonneg_true = 0;     % signal is not non-negative (only matters when xreal_true=1)
          run_fu = 0;   % don't run fienup: only works for orthogonal A
          
      case 4 % iid matrix 2x oversampled, real positive non-sparse signal, high SNR
          A_type = 'iid'; % one of 'iid', 'rdft', 'mdft', 'odft', 'o2dft', 'm2dft'
                        % ...see definitions in generatePRLinTrans.m
          SNRdB_true = 80;      % true SNR in dB. 
          nx = 512;     % # signal coefficients
          nz = 2*nx;    % # phaseless measurements 
          nk = nx;      % signal sparsity
          xreal_true = 1;       % real-valued signal
          xnonneg_true = 1;     % signal is non-negative (only matters when xreal_true=1)
          run_fu = 0;   % don't run fienup: only works for orthogonal A
          
      case 5 % 1D FFT, 2x oversampled, real sparse signal, high SNR
          A_type = 'odft'; % one of 'iid', 'rdft', 'mdft', 'odft', 'o2dft', 'm2dft'
                        % ...see definitions in generatePRLinTrans.m
          SNRdB_true = 100;     % true SNR in dB. 
          nx = 512;     % # signal coefficients
          nz = 2*nx;    % # phaseless measurements
          nk = 35;      % signal sparsity [35]
          xreal_true = 1;       % real-valued signal
          xnonneg_true = 0;     % signal is not non-negative (only matters when xreal_true=1)
          run_fu = 1;   % run fienup for comparison
          
      case 6 % 2D FFT, 1x oversampled, real sparse signal, high SNR
          A_type = 'o2dft'; % one of 'iid', 'rdft', 'mdft', 'odft', 'o2dft', 'm2dft'
                        % ...see definitions in generatePRLinTrans.m
          SNRdB_true = 100;     % true SNR in dB. 
          nx = 32^2;    % # signal coefficients
          nz = nx;      % # phaseless measurements
          nk = 50;      % signal sparsity [40]
          xreal_true = 1;       % real-valued signal
          xnonneg_true = 0;     % signal is not non-negative (only matters when xreal_true=1)
          run_fu = 1;   % run fienup for comparison
  end

  % set signal parameters (common to all scenarios)
  xmean0_true = 0*ones(nx,1);   % mean of non-zero signal coefficients
  xvar0_true = 1*ones(nx,1);    % variance of non-zero signal coefficients
  sparseRat_true = nk/nx;       % fraction of components of x that are non-zero

  % set simulation parameters (common to all scenarios)
  run_po_gamp = 1;      % run phase-oracle GAMP? (needs access to output phases)
  run_po_lmmse = 0;     % run phase-oracle LMMSE? (needs access to output phases)
  plot_est = 1;         % plot signal estimates? (0=no, #=figure number)
  plot_gamp = 2;        % plot gamp history? (0=no, #=figure number)
  plot_fu = 3;          % plot fienup history? ((0=no, #=figure number))

  % Set default phase-retrieval GAMP options
  clear opt_pr;
  clear opt_gamp_user;
  opt_pr.maxTry = 10; 
  opt_pr.xreal = xreal_true;            % =1 for real-valued x
  opt_pr.xnonneg = xnonneg_true;        % =1 for non-negative x (when xreal_true=1)
  opt_pr.sparseRat = sparseRat_true;    % use true sparsity rate
  opt_pr.sparseRatTry = sparseRat_true; % use true sparsity rate
  opt_pr.xmean0 = xmean0_true;          % use true mean
  opt_pr.xvar0 = xvar0_true;            % use true variance
  opt_pr.SNRdB = SNRdB_true;            % use true SNR 
  opt_pr.plot = plot_gamp;              % plot gamp history as indicated above

  % Set matrix-dependent customizations
  if strcmp(A_type,'odft')||strcmp(A_type,'o2dft'),
    opt_pr.maxTry = 100; 
    opt_pr.xreal = 0;
    opt_pr.sparseRat = 0.01*sparseRat_true; 
    opt_pr.sparseRatTry = sparseRat_true; 
    if isfield(opt_pr,'xvar0'), opt_pr=rmfield(opt_pr,'xvar0'); end; 
  end
  if strcmp(A_type,'o2dft')&&(xreal_true==0),
    error('disambiguation not implemented for complex 2Dfft!')
  end

  % Create a random Bernoulli-Gaussian signal vector
  if xreal_true,
    x0 = xmean0_true + sqrt(xvar0_true).*randn(nx,1);   % dense real-Gaussian
    if xnonneg_true,
      bad = find(x0<0);                         % bad indices
      while length(bad)>0,
        x0(bad) = xmean0_true(bad) + sqrt(xvar0_true(bad)).*randn(size(bad));% redraw 
        bad = find(x0<0);
      end;
    end;
  else
    x0 = xmean0_true + sqrt(xvar0_true/2).*(randn(nx,2)*[1;1i]);        % dense complex-Gaussian
  end;
  nk = round(nk);
  indx = randperm(nx);
  x = zeros(nx,1); x(indx(1:nk)) = x0(indx(1:nk));      % insert zeros

  % Create a random measurement matrix
  A = generatePRLinTrans(nz,nx,A_type,x);

  % Add complex-valued noise based on the specified SNR, and drop phase
  z = A*x;
  wvar = 10^(-SNRdB_true/10)*mean(abs(z).^2);
  w = sqrt(wvar/2)*(randn(nz,2)*[1;1i]);
  y = z + w;
  abs_y = abs(y);
  SNRdBhat = 20*log10(norm(z)/norm(w));

  %%%%%%%%%%%%%%%%%%%%%%%%%%
  % end demonstration mode %
  %%%%%%%%%%%%%%%%%%%%%%%%%%

 elseif nargin==1,

  error('must specify at least two input arguments')

 else,% nargin>=2

  if isa(A, 'double')
    A = MatrixLinTrans(A);
  end
  [nz,nx] = A.size;
  run_fu = 0;           
  plot_fu = 0;          
  plot_est = 0;         
  run_po_lmmse = 0;     % must be 0! (since no access to phase of observations)
  run_po_gamp= 0;       % must be 0! (since no access to phase of observations)

 end;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % load/set the phase-retrieval options %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 % real-valued signal? =1 for yes and =0 for no (complex-valued) [dflt:0]
 if exist('opt_pr')&&isfield(opt_pr,'xreal'), 
   if (opt_pr.xreal~=0)&&(opt_pr.xreal~=1),
     error('opt_pr.xreal must be either 0 or 1')
   end;
   xreal = opt_pr.xreal;
 else 
   xreal = 0;           % default is a complex-valued signal
 end;

 % non-negative signal? =1 for yes (only if real-valued) and =0 for no [dflt:0]
 if exist('opt_pr')&&isfield(opt_pr,'xnonneg'), 
   if (opt_pr.xnonneg~=0)&&(opt_pr.xnonneg~=1),
     error('opt_pr.xnonneg must be either 0 or 1')
   end;
   xnonneg = opt_pr.xnonneg;
 else 
   xnonneg = 0;         % default is a generic, possibly negative signal 
 end;

 % minimum number of tries
 if exist('opt_pr')&&isfield(opt_pr,'minTry'), 
   minTry = opt_pr.minTry;
 else 
   minTry = 1;          % default is minimum of 1 try 
 end;

 % maximum number of tries
 if exist('opt_pr')&&isfield(opt_pr,'maxTry'), 
   maxTry = opt_pr.maxTry;
   if (maxTry<minTry), warning('Setting maxTry=minTry!'); end
 else 
   maxTry = 10;         % default is maximum 10 tries (but more helps)
 end;
 maxTry = max(maxTry,minTry);

 % SNR in dB (used to set stopping tolerance for PR-GAMP, PO-GAMP, and Fienup 
 %            and xvar0 for PR-GAMP and PO-GAMP)
 if exist('opt_pr')&&isfield(opt_pr,'SNRdB'), 
   if ~isreal(opt_pr.SNRdB),
     error('opt_pr.SNRdB must be real')
   end;
   if opt_pr.SNRdB > 100
     warning('Setting SNRdB=100 to avoid numerical precision issues!')
   end;
   SNRdB = min(100,opt_pr.SNRdB);
 else 
   SNRdB = 60;          % default is 60 dB 
 end;
 SNR = 10^(SNRdB/10);
 wvar = (mean(abs_y.^2)/(SNR+1));  % used only for PO-LMMSE & PO-GAMP

 % initial SNR in dB used for EM iterations
 if exist('opt_pr')&&isfield(opt_pr,'SNRdBinit'), 
   if ~isreal(opt_pr.SNRdBinit),
     error('opt_pr.SNRdBinit must be real')
   end;
   if opt_pr.SNRdBinit > 100
     warning('Setting SNRdBinit=100 to avoid numerical precision issues!')
   end;
   SNRdBinit = min(100,opt_pr.SNRdBinit);
 else 
   SNRdBinit = 10;   % default is 10 dB 
 end;
 SNRinit = 10^(SNRdBinit/10);
 wvar_init = (mean(abs_y.^2)/(SNRinit+1));  % used by PR-GAMP

 % maximum value of SNR-adapted stopping tolerance
 if exist('opt_pr')&&isfield(opt_pr,'maxTol'), 
   if opt_pr.maxTol<0
     error('opt_pr.maxTol must be non-negative')
   end;
   maxTol = opt_pr.maxTol;
 else 
   maxTol = 1e-4;       % default is 1e-4
 end;

 % maximum number of EM iterations 
 if exist('opt_pr')&&isfield(opt_pr,'nitEM'), 
   if (opt_pr.nitEM~=round(opt_pr.nitEM))||(opt_pr.nitEM<0)
     error('opt_pr.nitEM must be a non-negative integer')
   end;
   nitEM = opt_pr.nitEM;
 else 
   nitEM = 20;          % default is 20 
 end;

 % signal sparsity rate in (0,1]
 if exist('opt_pr')&&isfield(opt_pr,'sparseRat'), 
   if (opt_pr.sparseRat<=0)||(opt_pr.sparseRat>1),
     error('opt_pr.sparseRat must be in (0,1]')
   end;
   sparseRat = opt_pr.sparseRat;
 else 
   sparseRat = 1;       % default is a non-sparse signal
 end;

 % mean of non-zero signal entries
 if exist('opt_pr')&&isfield(opt_pr,'xmean0'), 
   xmean0 = opt_pr.xmean0;
   if length(xmean0)==1, xmean0 = xmean0*ones(nx,1); end;
 else 
   xmean0 = zeros(nx,1);        % default is zero-mean  
 end;

 % variance of non-zero signal entries
 if exist('opt_pr')&&isfield(opt_pr,'xvar0'), 
   if any(~isreal(opt_pr.xvar0))||any(opt_pr.xvar0<0)||(sum(opt_pr.xvar0)==0),
     error('opt_pr.xvar0 must be non-negative with some positive entries')
   end;
   xvar0 = opt_pr.xvar0;
   if length(xvar0)==1, xvar0 = xvar0*ones(nx,1); end;
 else   % default is to estimate from measurements (based on SNR & sparseRat)
   A_fro_2 = sum(A.multSq(ones(nx,1)));
   xvar0 = (norm(abs_y)^2*SNR/(SNR+1)/sparseRat/A_fro_2)*ones(nx,1); 
 end;

 % determine whether or not to plot the GAMP history
 if exist('opt_pr')&&isfield(opt_pr,'plot'), 
   plot_gamp = opt_pr.plot;     % when non-zero, specifies plotting figure
 elseif nargin~=0, 
   plot_gamp = 0;               % default is to not plot the history 
 end;
 if plot_gamp>0, computeHist=1; end;

 % determine whether or not to print the residual
 if exist('opt_pr')&&isfield(opt_pr,'verbose'), 
   print_on = opt_pr.verbose;
 else 
   print_on = 1;                % default is to print the residual
 end;

 % normalized residual-error at which to declare success
 if exist('opt_pr')&&isfield(opt_pr,'nresStopdB'), 
   nresStopdB = opt_pr.nresStopdB;
 else 
   nresStopdB = -(SNRdB+2);     % default is 2dB better than SNR
 end;

 % maximum sparsity ratio tolerated to declare success
 if exist('opt_pr')&&isfield(opt_pr,'sparseRatTry'), 
   sparseRatTry = opt_pr.sparseRatTry;
 else 
   sparseRatTry = inf;          % default = inf (no constraint)
 end;

 % possible scaling of the initialization magnitude
 if exist('opt_pr')&&isfield(opt_pr,'init_gain'), 
   init_gain = opt_pr.init_gain;
 else 
   init_gain = 100;
%  init_gain = min(100,1/sparseRat^2)   % default gain on random initializations
                        % ...with sparse signals, a kick seems to be helpful 
 end;

 % Toggle adaptive stepsize
 if exist('opt_pr')&&isfield(opt_pr,'adaptStep'), 
   adaptStep = opt_pr.adaptStep;
 else 
   adaptStep = 1;       % default is to use adaptive stepsizes 
 end;

 % Toggle Bethe version of adaptive stepsize
 if exist('opt_pr')&&isfield(opt_pr,'adaptStepBethe'), 
   adaptStepBethe = opt_pr.adaptStepBethe;
 else 
   adaptStepBethe = 1;  % default is to use Bethe version of adaptive stepsize 
 end;

 % Choose type of EM learning
 if exist('opt_pr')&&isfield(opt_pr,'EMtype'), 
   EMtype = opt_pr.EMtype;
 else 
   %EMtype = 12;  % default is 12
   EMtype = 0; % default is 0
 end;

 %%%%%%%%%%%%%%%%%%%%%%%%
 % set the GAMP options %
 %%%%%%%%%%%%%%%%%%%%%%%%
 opt_gamp_init = GampOpt();             % load default options
 opt_gamp_init.legacyOut = 0;           % don't use legacy output format
 opt_gamp_init.nit = 200;                   % maximum number of iterations [dflt:200]
 opt_gamp_init.adaptStep = adaptStep;   % toggle adaptive stepsize
 opt_gamp_init.adaptStepBethe = adaptStepBethe;  % toggle Bethe version of adaptive stepsize
 opt_gamp_init.step = 0.25;                 % initial stepsize, <1 seems important [dflt:0.25]
 opt_gamp_init.stepMax = 0.25;          % max stepsize, <1 seems important [dflt:0.25]
 opt_gamp_init.stepMin = 0.05;          % min stepsize, >0 seems to help [dflt:0.05]
 opt_gamp_init.stepWindow = 50;         % adaptive stepsize window [dflt:50]
 opt_gamp_init.stepIncr = 1.1;          % step incr when successful [dflt:1.1]
 opt_gamp_init.stepDecr = 0.2;          % step decr when unsuccessful [dflt:0.2-0.4]
 opt_gamp_init.varNorm = 1;                 % seems important to be on [dflt:1]
 opt_gamp_init.pvarStep = 1;            % seems important to be on [dflt:1]
 opt_gamp_init.rvarStep = 0;            % seems important to be off [dflt:0]
 opt_gamp_init.uniformVariance = 0;     % depends on matrix type [dflt:0] 
 opt_gamp_init.histIntvl = 1;           % setting >1 saves memory [dflt:1]

 % rewrite those options for which the user has specified custom values
 if exist('opt_gamp_user')
   if isstruct(opt_gamp_user)                   % the user specified a structure
     fields = fieldnames(opt_gamp_user);
   elseif ismethod(opt_gamp_user,'GampOpt')     % the user specified a GampOpt object
     fields = properties(opt_gamp_user);
   else
     error('the 4th input is of an unrecognized type')
   end;
   for i=1:size(fields,1),                      % change specified fields
     opt_gamp_init = setfield(opt_gamp_init,fields{i},getfield(opt_gamp_user,fields{i}));
   end;
 end;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % Create an input estimation class (Bernoulli-Gaussian) %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 if xreal
   if xnonneg
     inputEst0 = NNGMEstimIn(ones(nx,1),xmean0, xvar0);
   else
     inputEst0 = AwgnEstimIn(xmean0, xvar0);
   end
 else
   inputEst0 = CAwgnEstimIn(xmean0, xvar0);
 end;
 if sparseRat < 1, 
   inputEst = SparseScaEstim(inputEst0,sparseRat);
 else
   inputEst = inputEst0;
 end;
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % if desired, compute the phase-oracle LMMSE solution %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 if run_po_lmmse&&(nargin==0)&&print_on,
   xmean = sparseRat*xmean0;
   xvar = sparseRat*xvar0;
   I = eye(nx); AA = NaN*ones(nz,nx); for i=1:nx, AA(:,i) = A.mult(I(:,i)); end;
   xhatLMMSE = xmean + xvar.*(AA'*((AA*diag(xvar)*AA'+diag(wvar*ones(nz,1)))\(y-AA*xmean)));
   nmseLMMSE = 20*log10( norm(x-xhatLMMSE)/norm(x) );
   fprintf(1,'LMMSE: MSE = %5.1f dB\n', nmseLMMSE);
 end;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % if desired, run the Fienup algorithm %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 if run_fu&&(print_on|plot_fu),
   optFU.maxTry = 200;
   optFU.nresStopdB = 10^(-(SNRdB+2)/10);
   optFU.nit = 1000; % maximum # iterations
   optFU.tol = min(10^(-SNRdB/10),1e-4); % stopping tolerance
   optFU.xreal = isreal(x(1));
   if strcmp(A_type,'odft')||strcmp(A_type,'o2dft')
     Afu = @(x) A.mult(x);
     AfuInv = @(z) A.multTr(z);
   else
     error('Fienup not implemented for that A_type')
   end
   [xhatFU,numTryFU,xhatFUHist] = fienupEst(abs_y,nx,Afu,AfuInv,nk,optFU);
   nresFU = 20*log10(norm(abs(A*xhatFU)-abs_y)/norm(abs_y));
   if print_on,
     if nargin==0,
       if strcmp(A_type,'odft')
         [xhatFU,disamvec] = disambig1Dfft(xhatFU,x); disammat = disamvec;
       elseif strcmp(A_type,'o2dft')
         [xhatFU,disamvec,disammat] = disambig2Drfft(xhatFU,x,sqrt(nz),sqrt(nz));
       end 
       nmseFU = 20*log10(norm(x-xhatFU)/norm(x));
       fprintf(1,'FU:    NMSE = %5.1f dB, NRES = %5.1f dB (after %d tries)\n', [nmseFU,nresFU,numTryFU]);
     else
       fprintf(1,'FU:    RES = %5.1f dB (after %d tries)\n', [nresFU,numTryFU]);
     end;
   end;
   % plot the Fienup history
   if plot_fu,
     figure(plot_fu); clf;
     nit = size(xhatFUHist,2);
     if nit==1, xhatFUHist = xhatFUHist*[1,1]; nit=2; end;
     zhatFUHist = zeros(nz,nit); 
     for iter=1:nit, zhatFUHist(:,iter) = A*xhatFUHist(:,iter); end;
     nresFU_ = 10*log10(sum((abs(zhatFUHist) ...
                                -abs_y*ones(1,nit)).^2,1)/norm(y)^2);
     if nargin==0,
       nmseFU_ = 10*log10(sum(abs(x*ones(1,nit) ...
                        -disammat(xhatFUHist)).^2,1)/norm(x)^2);
       plot([nmseFU_;nresFU_].'); 
       legend('error','residual')
       ylabel('dB')
     else
       plot(nresFU_); 
       ylabel('residual [dB]')
     end;
     xlabel('iter')
     title(['FU (after ',num2str(numTryFU),' tries)'])
     grid on
     drawnow
   end;% plot_fu
 end;% run_fu

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % if desired, compute the phase-oracle GAMP solution %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 if run_po_gamp&&(nargin==0)&&print_on,
   opt_po = GampOpt();
   opt_po.legacyOut = 0;
   opt_po.stepWindow = 25;
   opt_po.adaptStep = 1; % needed for oscillations with very small nk
   opt_po.adaptStepBethe = 1; % although old method works better for nx=nz=nk
   opt_po.stepMax = 1.0;
   opt_po.stepDecr = 0.5;
   opt_po.stepIncr = 1.1;
   opt_po.stepMin = 0.25; % needed only for nx=nz=nk
   opt_po.nit = 1e5; % large setting needed for nx=nz=nk
   opt_po.varNorm = 0; % doesn't really matter
   opt_po.tol = min(1e-4,10^(-SNRdB/10));       
   outputEstPO = CAwgnEstimOut(y, wvar*ones(nz,1));
   if computeHist 
     [estFinPO,optFinPO,estHistPO] = gampEst(inputEst, outputEstPO, A, opt_po);
     %figure(1); clf; gampShowHist(estHistPO,optFinPO,x,z)
   else
     [estFinPO,optFinPO] = gampEst(inputEst, outputEstPO, A, opt_po);
   end
   xhatPO = estFinPO.xhat;
   nmsePOdB = 20*log10( norm(x-xhatPO)/norm(x) );
   nresPOdB = 20*log10( norm(abs_y-abs(A*xhatPO))/norm(abs_y) );
   fprintf(1,'PO:    NMSE = %5.1f dB, NRES = %5.1f dB\n', nmsePOdB,nresPOdB);
 end;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % Run phase-retrieval GAMP, with multiple re-tries if needed %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 nresGAMPdB_best = inf;
 for t=1:maxTry,

   % initialize GAMP
   if (t==1)&&(exist('opt_pr')&&isfield(opt_pr,'xhat0')), % user-supplied init
     xinit = opt_pr.xhat0;
     opt_gamp_init.xhat0 = xinit;               
   else % random initialization
      if xreal,
       xinit = sparseRat*xmean0 + sqrt(sparseRat*xvar0).*randn(nx,1);
      else
       xinit = sparseRat*xmean0 + sqrt(sparseRat*xvar0/2).*(randn(nx,2)*[1;1i]);
      end;
     %xinit = A.multTr(abs_y.*sign(A.mult(xinit))); % only makes sense for FFTs
     opt_gamp_init.xhat0 = init_gain*xinit;             % notice init_gain!
   end;
   opt_gamp_init.xvar0 = mean(abs(opt_gamp_init.xhat0-sparseRat*xmean0).^2)*ones(nx,1);

   
   % prepare for EM iterations
   wvar_hat = [wvar_init,nan(1,nitEM+1)];
   SNRdB_hat = nan(1,nitEM+1);
   if computeHist, estHist_ = []; end;
   nresGAMPdB = nan(1,nitEM+1);
   nmseGAMPdB = nan(1,nitEM+1);
   xhat_old = inf(nx,1);
   support_size = nan;
   nit_ = nan(1,nitEM+1);
   
   % loop over EM iterations
   for em=1:nitEM+1
       
     % setup for current EM iteration
     SNRdB_hat(em) = 10*log10( mean(abs_y.^2)./wvar_hat(em) );
     tolGAMP = min(10^(-SNRdB_hat(em)/10),maxTol);    % GAMP stopping tolerance
     tolEM = 10*tolGAMP; % EM stopping tolerance [dflt: 10*tolGAMP]
     if em==1
       opt_gamp = opt_gamp_init;
       opt_gamp.tol = tolGAMP;
       opt_gamp.xvar0auto = 0; % auto-set xvar0 on first EM iteration? [dflt=0]
       outputEst = ncCAwgnEstimOut(abs_y,wvar_hat(em)*ones(nz,1),0,0); 
     else %em>1
       opt_gamp = optFin.warmStart(estFin,'tol',tolGAMP);
       opt_gamp.xvar0auto = false; % never do this on later EM iterations!
       if (EMtype==0) 
         if (em==2)
           % turn on internal estimation of noise variance 
           outputEst = ncCAwgnEstimOut(abs_y,wvar_hat(em)*ones(nz,1),0,1); 
         end
       else
         % use externally estimated noise variance
         outputEst = ncCAwgnEstimOut(abs_y,wvar_hat(em)*ones(nz,1),0,0); 
       end
     end
       
     % run GAMP
     if computeHist
       [estFin,optFin,estHist] = gampEst(inputEst, outputEst, A, opt_gamp);
       estHist_ = appendEstHist(estHist_,estHist);
     else
       [estFin,optFin] = gampEst(inputEst, outputEst, A, opt_gamp);
     end
     nresGAMPdB(em) = 20*log10(norm(abs(A*estFin.xhat)-abs_y)/norm(abs_y));

     % disambiguate
     if nargin==0
       if strcmp(A_type,'odft')
         [xhat,disamvec] = disambig1Dfft(estFin.xhat,x);
         disammat = disamvec;
       elseif strcmp(A_type,'o2dft')
         [xhat,disamvec,disammat] = disambig2Drfft(estFin.xhat,x,sqrt(nz),sqrt(nz));
       else
         angl = sign(estFin.xhat'*x);
         disamvec = @(xhat) xhat*angl; 
         disammat = disamvec;
         xhat = disamvec(estFin.xhat);
       end
     end   
   
     % print GAMP result
     if print_on,
       if em==1, fprintf(1,'\nAttempt number %d:\n',t); end;
       if nargin==0,
         nmseGAMPdB(em) = 20*log10(norm(x-xhat)/norm(x));
         fprintf(1,'%2d GAMP: NMSE = %5.1f dB, NRES = %5.1f dB\n', [em,nmseGAMPdB(em),nresGAMPdB(em)]);
       else
         fprintf(1,'%2d GAMP: NRES = %5.1f dB\n', [em,nresGAMPdB(em)]);
       end;
     end;
     
     % check for misconvergence
     isnonzero = (norm(estFin.rhat)^2 > 1e-6*sparseRat*sum(abs(xmean0).^2+xvar0));
     if isnan(nresGAMPdB(em))||isinf(nresGAMPdB(em))||~isnonzero
       % try a few things that might help prevent misconvergence
       init_gain = init_gain*2; 
       opt_gamp_init.stepWindow = ceil(opt_gamp_init.stepWindow*0.75);  % decrease adaptive stepsize window
       break;
     end

     % check EM stopping criterion
     if (norm(xhat_old-estFin.xhat)/norm(estFin.xhat) < tolEM )||(em==nitEM+1) 
       break;
     end;
     xhat_old = estFin.xhat;
  
     % update noise variance estimate
     switch EMtype
       case 0
         wvar_hat(em+1) = mean(outputEst.var0); % extract internally estimated noise variance
       case 12
         wvar_hat(em+1) = mean(2*(abs_y-abs(A.mult(estFin.xhat))).^2);
       otherwise
         error('Unknown EMtype')
     end
     nit_(em) = estFin.nit;

   end % for em=1:nitEM+1

   % trim EM outputs
   wvar_hat = wvar_hat(1:em);
   nresGAMPdB = nresGAMPdB(1:em);
   nmseGAMPdB = nmseGAMPdB(1:em);
   SNRdB_hat = SNRdB_hat(1:em);
   it_switch = cumsum(nit_(1:em-1))+0.5;

   % plot GAMP history
   if plot_gamp,
     figure(plot_gamp); if (t==1)&&(em==1), clf; end;
     nit = size(estHist_.xhat,2);
     zhat = zeros(nz,nit); 
     for iter=1:nit, zhat(:,iter) = A*estHist_.xhat(:,iter); end;
     nresGAMPdB_ = 10*log10(sum((abs(zhat) ...
                        -abs_y*ones(1,nit)).^2,1)/norm(abs_y)^2);
     subplot(411)
      if nargin==0,
        nmseGAMPdB_ = 10*log10(sum(abs(x*ones(1,nit) ...
                        -disammat(estHist_.xhat)).^2,1)/norm(x)^2);
        plot(estHist_.it,nmseGAMPdB_,estHist_.it,nresGAMPdB_); 
        legend('signal error','residual error')
        ylabel('dB')
      else
        plot(nresGAMPdB_); 
        ylabel('residual [dB]')
      end;
      axe = axis;
      axis([0,nit,axe(3:4)])
      hold on; 
        for i=1:em-1, plot(it_switch(i)*[1,1],axe(3:4),'k-'); end; 
      hold off;
      title(['PR-GAMP3 (try #',num2str(t),')'])
      grid on;
     subplot(412)
      plot(estHist_.it,10*log10(mean(estHist_.xvar,1)),...
         estHist_.it,10*log10(mean(estHist_.pvar,1)),...
         estHist_.it,10*log10(mean((1/nx)*abs(estHist_.rhat).^2,1)));
      legend('xvar','pvar','rhat\^2')
      axe = axis;
      axis([0,nit,axe(3:4)])
      hold on; 
        for i=1:em-1, plot(it_switch(i)*[1,1],axe(3:4),'k-'); end; 
      hold off;
      ylabel('dB')
      grid on
     subplot(413)
      if (opt_gamp_init.adaptStep)
        if 1 % plot normalized cost
          cost = -estHist_.val-min(-estHist_.val)+1; % force min{cost} = 1
          semilogy(estHist_.it,cost,'.-',estHist_.it(~estHist_.pass),cost(~estHist_.pass),'r.'); 
          axe=axis;
          axis([0,nit,axe(3),max(cost)])
          ylabel('cost')
        else % plot raw val
          plot(estHist_.it,estHist_.val,'.-',estHist_.it(~estHist_.pass),estHist_.val(~estHist_.pass),'r.'); 
          axe=axis;
          ylabel('val')
        end
        hold on; 
          for i=1:em-1, plot(it_switch(i)*[1,1],axe(3:4),'k-'); end; 
        hold off;
        grid on;
      else
        cla;
      end
     subplot(414)
      handy=plot(estHist_.it,estHist_.step,'.-',estHist_.it(~estHist_.pass),estHist_.step(~estHist_.pass),'r.');
      hold on; plot(estHist_.it,estHist_.stepMax,'g:'); hold off;
      set(handy,'MarkerSize',8)
      axe = axis; axis([0,nit,0.9*min(estHist_.step),1.1*max(estHist_.step)])
      axe = axis;
      hold on; 
        for i=1:em-1, plot(it_switch(i)*[1,1],axe(3:4),'k-'); end; 
      hold off;
      ylabel('step')
      xlabel('iter')
      grid on;
      drawnow
   end; % plot_gamp

   % plot signal estimates
   if plot_est,
     figure(plot_est); clf;

     if isreal(estFin.xhat), simp=@(x) x; else simp=@(x) abs(x); end;
     if nargin==0,
       handy = stem(simp(xhat));
       leg_str = strvcat('GAMP');
       hold on;
       if run_po_lmmse,
         handy = [handy; stem(simp(xhatLMMSE),'sc--')];
         leg_str = strvcat(leg_str,'PO-LMMSE');
       end;
       if run_po_gamp,
         handy = [handy; stem(simp(xhatPO),'+k--')];
         leg_str = strvcat(leg_str,'PO-GAMP');
       end;
       if run_fu,
         handy = [handy; stem(simp(xhatFU),'vg--')];
         leg_str = strvcat(leg_str,'FU');
       end;
       handy = [handy; stem(simp(x),'xr--')];
       leg_str = strvcat(leg_str,'true');
       hold off;
       ylabel('absolute value')
       xlabel('signal index')
     else
       handy = stem(simp(estFin.xhat));
       leg_str = strvcat('GAMP');
     end;
     legend(handy,leg_str);
     drawnow
   end; % plot_est

   % remember result if yields lowest residual so far
   if (nresGAMPdB(em)<nresGAMPdB_best)
     nresGAMPdB_best = nresGAMPdB(em);
     xinit_best = xinit;
     estFin_best = estFin;
     optFin_best = optFin;
     wvar_best = wvar_hat;
     SNRdB_best = SNRdB_hat;
     nmseGAMPdB_best = nmseGAMPdB;
     if computeHist, estHist_best = estHist_; end
   end

   % finish if sufficiently low residual
   if (t>=minTry)&&(nresGAMPdB_best<nresStopdB)

     % compute posterior support probabilities
     if xreal
       post_prob = 1./(1+ (1-sparseRat)/sparseRat*exp(...
        -0.5*abs(estFin_best.rhat).^2./estFin_best.rvar ...
        +0.5*abs(estFin_best.rhat-xmean0).^2./(estFin_best.rvar+xvar0) )...
        .*sqrt((estFin_best.rvar+xvar0)./estFin_best.rvar) );
     else
       post_prob = 1./(1+ (1-sparseRat)/sparseRat*exp(...
        -abs(estFin_best.rhat).^2./estFin_best.rvar ...
        +abs(estFin_best.rhat-xmean0).^2./(estFin_best.rvar+xvar0) )...
        .*(estFin_best.rvar+xvar0)./estFin_best.rvar );
     end
     support_size = sum(post_prob>0.5);

     % really finish if sufficiently small support
     if (support_size <= round(sparseRatTry*nx))
       break; % support looks good, so finish 
     else
       nresGAMPdB_best = inf; % forget this "best" residual
     end 
   end % if

 end;% for t=1:maxTry;
 numTry = t;
   
 %%%%%%%%%%%%%%%%%
 % write outputs %
 %%%%%%%%%%%%%%%%%
 xhat = estFin_best.xhat;
 clear out;
 out.xmean0 = xmean0;
 out.xvar0 = xvar0;
 out.wvar = wvar_best;
 out.SNRdB_est = SNRdB_best;
 out.numTry = numTry;
 out.support_size = support_size;

 if nargin==0
   % report accuracy of EM noise-variance estimate
   wvarHat_over_wvarTrue_dB = 10*log10(wvar_best/wvar)
 end
