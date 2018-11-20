% prGAMP Phase-Retrieval Generalized Approximate Message Passing 
%
%   prGAMP uses Sundeep Rangan's GAMP algorithm to perform phase 
%   retrieval, i.e., to estimate the length-N signal vector x given 
%   knowledge of A and abs(y), in the additive linear model 
%       y = A.mult(x) + w,
%   Here, the noise is assumed to be independent zero-mean Gaussian 
%   and the signal is assumed to be Bernoulli-Gaussian.
%
%   This code is a part of the GAMPmatlab package.
%
% SYNTAX: 
%
%   [xhat,out,estHist] = prGAMP(abs_y, A, opt_pr, opt_gamp)
%
% DEMO MODE: 
%
%   run without any inputs to see a demo.
%
% STANDARD OUTPUTS:
%
%   xhat  : estimated length-N signal such that abs_y ~= abs(A*xhat)
%
%   estHist : GAMPmatlab structure detailing algorithm evolution 
%
%   out.xvar0: prior signal variance
%   out.threshTry : normalized residual level in dB needed to avoid re-initialization
%   out.snrMax : schedule of SNR-upper-bound
%   out.tol : normalized stopping tolerance
%
% STANDARD INPUTS:
%
%   abs_y : observed length-M vector of magnitudes 
%
%   A     : GAMPmatlab linear-transform class
%
% OPTIONAL INPUTS:
%
%   opt_pr.xreal : 1 if signal vector is real-valued, and 0 otherwise (default=0)
%   opt_pr.xnonneg : 1 if signal vector is non-negative, and 0 otherwise (default=0)
%   opt_pr.sparseRat : signal sparsity rate in (0,1] (default=1)
%   opt_pr.xmean0 : length-N signal mean vector (default=0)
%   opt_pr.xvar0 : length-N signal variance vector (default via abs_y, ||A||_F, xmean0)
%   opt_pr.xhat0 : length-N signal initialization (default is random)
%   opt_pr.maxTry : integer number of re-initialization attempts (default=10)
%   opt_pr.plot : plot the evolution of the residual, cost, and stepsize? (default=0)
%   opt_pr.print : print the residual? (default=1)
%
%   opt_gamp : do not include this unless you really know what you are doing!!

function [xhat,out,estHist_] = prGAMP(abs_y,A,opt_pr,opt_gamp_user)
%nargin=0;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % load or generate signals %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
 if nargin==0,

  %Handle random seed
  if verLessThan('matlab','7.14')
    defaultStream = RandStream.getDefaultStream;
  else
    defaultStream = RandStream.getGlobalStream;
  end;
  if 0
    savedState = defaultStream.State;
    save random_state.mat savedState;
  else
    load random_state.mat
  end
  defaultStream.State = savedState;

  % Simulation Parameters
  run_pa_gamp= 0;% run phase-aware GAMP? (needs access to output phases)
  run_pa_lmmse = 0;% run phase-aware LMMSE? (needs access to output phases)
  run_gs = 0;	% run gerchberg-saxon?
  plot_est = 1;	% plot signal estimates?
  plot_gamp = 2;% plot gamp evolution?
  plot_gs = 3;	% plot gerchberg-saxon trajectory? (only applies when run_gs=1)

  % Signal Parameters
  xreal = 1;	% real-valued x?
  xnonneg = 1;	% non-negative x?  (only applies when xreal=1)
  nx = 32^2;	% Number of input components (dimension of x)
  nz = ceil(nx);	% Number of output components (dimension of y)
  		% ...if xreal=1, then "compressive" means nz<nx
  		% ...if xreal=0, then "compressive" means nz<2*nx
		% ...without sparsity, need ~2*nx for real, ~4*nx for cmplx
  nk = ceil(nx/16);	% Number of nonzero input components (sparsity of x) 
  snr = 50;	% SNR in dB. 
  A_type = 'rdft';	% one of 'iid', 'rdft', 'mdft', 'odft', 'o2dft', 'm2dft'

  %nx = nz; nk = sqrt(nx);
  %nx = 64; nk = 5; nz = 75;	% Fig. 4 from Ohlsson
  %nx = 64; nk = 2; nz = 40; xreal=0; snr = 100;	% allerton submission table 3a
  %nx = 32; nk = 2; nz = 20; xreal=0; snr = 100;	% allerton submission table 3b

  % Create a random Bernoulli-Gaussian vector
  xmean0 = 1*ones(nx,1);
  xvar0 = 1*ones(nx,1);
  if xreal,
    x0 = xmean0 + sqrt(xvar0).*randn(nx,1);	% dense real-Gaussian
    if xnonneg,
      bad = find(x0<0);				% bad indices
      while length(bad)>0,
        x0(bad) = xmean0(bad) + sqrt(xvar0(bad)).*randn(size(bad));% redraw 
        bad = find(x0<0);
      end;
    end;
  else
    x0 = xmean0 + sqrt(xvar0/2).*(randn(nx,2)*[1;1i]);	% dense complex-Gaussian
  end;
  nk = round(nk);
  sparseRat = nk/nx;	% fraction of components of x that are non-zero
  indx = randperm(nx);
  x = zeros(nx,1); x(indx(1:nk)) = x0(indx(1:nk));	% insert zeros

  % Create a random measurement matrix
  if strcmp(A_type,'iid'),				% iid gaussian :)
    A = (1/sqrt(2*nx))*(randn(nz,nx)+1i*randn(nz,nx));
    A = MatrixLinTrans(A);
  elseif strcmp(A_type,'rdft'),				% iidN after DFT :)
    F = dftmtx(nx)/sqrt(nx);
    A = (1/sqrt(2*nx))*(randn(nz,nx)+1i*randn(nz,nx))*F;
    A = MatrixLinTrans(A);
  elseif strcmp(A_type,'mdft'),				% DFT after mask :)
    F = dftmtx(nx)/sqrt(nx);
    A = F;			% first mask is trivial
    for i=1:ceil(nz/nx)-1,
      h = round(rand(nx,1));    % other masks are random binary 
      A = [A;F*diag(h)]; 
    end;
    A = A(1:nz,:);		% trim size
    A = MatrixLinTrans(A);
  elseif strcmp(A_type,'odft'),				% oversampled DFT :(
    F = dftmtx(nz)/sqrt(nx);
    A = F(:,1:nx);
    [dum,SampLocs]=unique(abs(A*x));	% trim non-unique magnitudes
    Arank = rank(A(sort(SampLocs),:));
    if Arank<nx, 
      warning(['recovery impossible: linear transform rank=',...
      	num2str(Arank),' while nx=',num2str(nx),'.']); 
    end;
    A = MatrixLinTrans(A(sort(SampLocs),:));
    nz = length(SampLocs);
  elseif strcmp(A_type,'o2dft'),			% oversampled 2DFT :|
    nX = sqrt(nx);
    ndft = sqrt(nz);
    if (ceil(nX)~=nX)||(ceil(ndft)~=ndft), 
      error('nX or ndft is not an integer'); 
    end;
    A = sampTHzLinTrans(nX,nX,ndft,ndft);
    [dum,SampLocs]=unique(abs(A.mult(x)));	% trim non-unique magnitudes
    A = sampTHzLinTrans(nX,nX,ndft,ndft,sort(SampLocs));
    nz = length(SampLocs);
  elseif strcmp(A_type,'m2dft'),			% 2DFT after masks :)
    num_mask_min = 4;	% minimum # of masks [dflt>=3]
    nX = sqrt(nx);
    if ceil(nX)~=nX,
      error('nX is not an integer'); 
    end;
    ndft = sqrt(2^(2*ceil(log2(nx)/2)));  	% ndft^2 >= nx, ndft=power-of-2
    num_masks = max(num_mask_min,ceil(nz/ndft^2));	
    if 1,					% random binary
      Masks = round(rand(nX,nX,num_masks));
      Masks(:,:,num_masks) = 1-Masks(:,:,1);	% ensure an invertible transform
    else					% shuffled non-binary
      mask_gains = rand(1,num_masks);		
      Masks = reshape( reshape(ones(nX,nX,1),nX^2,1)*(mask_gains(:).'),nX,nX,num_masks );
      for i=1:nX, for j=1:nX, Masks(i,j,:) = Masks(i,j,randperm(num_masks)); end; end;
    end;
    num_samp = ceil(nz/num_masks);
    if 1	% top left corner in Fourier space :)
      corner = zeros(ndft); 
      corner(1:ceil(sqrt(num_samp)),1:ceil(sqrt(num_samp))) = ...
      	ones(ceil(sqrt(num_samp)));
      indices = find(corner==1); 	
      SampLocs = indices(1:num_samp);
    else 	% random indices :(
      SampLocs = nan(num_samp,num_masks);
      for k=1:num_masks,
        indices = randperm(ndft^2).';	
        SampLocs(:,k) = sort(indices(1:num_samp));
      end;
    end;
    A = sampTHzLinTrans(nX,nX,ndft,ndft,SampLocs,Masks);
    nz = A.size;	% nz may have been rounded up
  else
    error('invalid A_type')
  end;

  % Compute the noise level based on the specified SNR. 
  wvar = 10^(-0.1*snr)*mean(abs(x).^2)*ones(nz,1);

  % Generate the noise 
  w = sqrt(wvar/2).*(randn(nz,2)*[1;1i]);
  z = A.mult(x);
  y = z + w;
  abs_y = abs(y);
  snr_hat = 20*log10(norm(z)/norm(w));

  % Set phase-retrieval GAMP options
  clear opt_pr;
  opt_pr.xreal = xreal;
  opt_pr.xnonneg = xnonneg;
  opt_pr.sparseRat = sparseRat;
  opt_pr.xmean0 = xmean0;
  opt_pr.xvar0 = xvar0;
  opt_pr.wvar = wvar;

 elseif nargin==1,

  error('must specify at least two input arguments')

 else,% nargin>=2

  [nz,nx] = A.size;
  run_gs = 0;		% can be either 0 or 1
  plot_gs = 0;		% can be either 0 or 1
  plot_est = 0;		% can be either 0 or 1
  run_pa_lmmse = 0;	% must be 0! (since no access to phase of observations)
  run_pa_gamp= 0;	% must be 0! (since no access to phase of observations)

 end;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % load/set the phase-retrieval options %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % The following are assumptions about the signal...
 % Real-valued? =1 for yes and =0 for no (complex-valued) [dflt:0]
 if exist('opt_pr')&&isfield(opt_pr,'xreal'), 
   if (opt_pr.xreal~=0)&&(opt_pr.xreal~=1),
     error('opt_pr.xreal must be either 0 or 1')
   end;
   xreal = opt_pr.xreal;
 else 
   xreal = 0;		% default is a complex-valued signal
 end;
 if exist('opt_pr')&&isfield(opt_pr,'xnonneg'), 
   if (opt_pr.xnonneg~=0)&&(opt_pr.xnonneg~=1),
     error('opt_pr.xnonneg must be either 0 or 1')
   end;
   xnonneg = opt_pr.xnonneg;
 else 
   xnonneg = 0;		% default is a possibly negative signal 
 end;
 % sparsity rate in (0,1]
 if exist('opt_pr')&&isfield(opt_pr,'sparseRat'), 
   if (opt_pr.sparseRat<=0)||(opt_pr.sparseRat>1),
     error('opt_pr.sparseRat must be in (0,1]')
   end;
   sparseRat = opt_pr.sparseRat;
 else 
   sparseRat = 1;	% default is a non-sparse signal
 end;
 % signal mean vector
 if exist('opt_pr')&&isfield(opt_pr,'xmean0'), 
   xmean0 = opt_pr.xmean0;
   if length(xmean0)==1, xmean0 = xmean0*ones(nx,1); end;
 else 
   xmean0 = zeros(nx,1);% default is zero-mean	
 end;
 % signal variance vector
 if exist('opt_pr')&&isfield(opt_pr,'xvar0'), 
   if any(~isreal(opt_pr.xvar0))||any(opt_pr.xvar0<=0),
     error('opt_pr.xvar0 must be positive')
   end;
   xvar0 = opt_pr.xvar0;
   if length(xvar0)==1, xvar0 = xvar0*ones(nx,1); end;
 else 			% default is to calculate variance as follows
   A_fro_2 = sum(A.multSq(ones(nx,1)));
   xvar0 = (norm(abs_y)^2 - sparseRat*norm(A.mult(xmean0))^2 ...
   	)/(sparseRat*A_fro_2)*ones(nx,1); % warning: this is a hack!
   min_stdv_to_mean = 0.1;	% minimum allowed stdv-to-mean ratio
   stdv_to_mean = max(sqrt(xvar0)/mean(abs(xmean0)),0);
   if stdv_to_mean < min_stdv_to_mean, 
     warning(['Setting ratio of estimated-stdv to |mean| at min value, ',...
     	num2str(min_stdv_to_mean)]);
     xvar0 = ones(nx,1)*(min_stdv_to_mean*mean(abs(xmean0)))^2;
   end;
 end;

 % The following is the noise assumption 
 if exist('opt_pr')&&isfield(opt_pr,'wvar'), 
   if any(~isreal(opt_pr.wvar))||any(opt_pr.wvar<0),
     error('opt_pr.wvar must be non-negative')
   end;
   wvar = opt_pr.wvar;
 else 
   wvar = zeros(nz,1);	% default is no noise 
 end;
 snr = 10*log10( mean(abs_y.^2) / mean(wvar) );	% SNR calculated from wvar

 % We can run GAMP several times under a list of specified maximum SNRs.
 % Keeping the first value sufficiently small seems to aid with convergence.
 % The last value essentially determines the final precision of the algorithm.
 if exist('opt_pr')&&isfield(opt_pr,'snrMax'), 
   if any(~isreal(opt_pr.snrMax)),
     error('opt_pr.snrMax must be real')
   end;
   snrMax = opt_pr.snrMax;
 else 
   %-------------------------------------
   snrMax = [25,50,inf,inf,inf,inf];	% default max-SNR schedule [dflt=25,50,inf]
% really we should change this to snr_gamp
   %-------------------------------------
 end;
 wvar_min = mean(abs_y.^2)./(10.^(snrMax/10));	% min-noise-variance schedule
tol = 10.^(-min(snrMax,snr)/10);		% schedule for stopping tolerance
% really we should change this to a scalar

 % Should we plot the GAMP evolution?
 if exist('opt_pr')&&isfield(opt_pr,'plot'), 
   plot_gamp = opt_pr.plot;	% when non-zero, specifies plotting figure
 elseif nargin~=0, 
   plot_gamp = 0;		% default is to not plot the evolution 
 end;

 % Should we print the residual?
 if exist('opt_pr')&&isfield(opt_pr,'print'), 
   print_on = opt_pr.print;
 else 
   print_on = 1;	% default is to print residual
 end;

 % We'll run GAMP from different random initializations until the residual 
 % level (in dB) is less than threshTry
 if exist('opt_pr')&&isfield(opt_pr,'threshTry'), 
   threshTry = opt_pr.threshTry;
 else 
   threshTry = -min(snr,snrMax(1));  	% dB-residual level defined as success
 end;
 if exist('opt_pr')&&isfield(opt_pr,'maxTry'), 
   maxTry = opt_pr.maxTry;
 else 
   %-------------------------------------
   maxTry = 10;		% default maximum number of initializations [dflt:10]
   %-------------------------------------
 end;

 % Scaling the initialization seems to aid convergence 
 if exist('opt_pr')&&isfield(opt_pr,'init_gain'), 
   init_gain = opt_pr.init_gain;
 else 
   %-------------------------------------
   init_gain = min(100,1/sparseRat^2);	% default gain on random initializations [dflt:100]
   			% ...with sparse signals, a kick seems to be helpful 
   %-------------------------------------
 end;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % set the GAMP options ... proper settings are important!! %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 opt_gamp = GampOpt(); 	% load default set
 opt_gamp.tol = tol(1);		% stopping tolerance 
opt_gamp.nit = 250; 		% maximum number of iterations [dflt:250]
opt_gamp.step = 0.25; 		% initial stepsize, <1 seems important [dflt:0.25]
opt_gamp.stepMax = 0.25; 	% max stepsize, <1 seems important [dflt:0.25]
 opt_gamp.stepWindow = 25;	% stepsize window [dflt:25]
 opt_gamp.stepIncr = 1.1;	% step incr when successful [dflt:1.1]
 opt_gamp.stepDecr = 0.2;	% step decr when unsuccessful [dflt:0.2-0.4]
 opt_gamp.varNorm = 1;		% seems important to be on [dflt:1]
 opt_gamp.pvarStep = 1;		% seems not too important [dflt:1]
 opt_gamp.uniformVariance = 0; 	% depends on matrix type [dflt:0]  
 opt_gamp.xvarMin = 1e-8;	% helps with numerical precision of NNGMEstimIn

 % rewrite those options for which the user has specified values
 if nargin>=4
   if isstruct(opt_gamp_user)			% the user specified a structure
     fields = fieldnames(opt_gamp_user);
   elseif ismethod(opt_gamp_user,'GampOpt')	% the user specified a GampOpt object
     fields = properties(opt_gamp_user);
   else
     error('the 4th input is of an unrecognized type')
   end;
   for i=1:size(fields,1),			% change specified fields
     opt_gamp = setfield(opt_gamp,fields{i},getfield(opt_gamp_user,fields{i}));
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

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % Create an output estimation class (noncoherent complexGaussian) %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 outputEst = ncCAwgnEstimOut(abs_y,max(wvar,wvar_min(1)*ones(nz,1))); % note variance limiting

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % if desired, compute the phase-aware LMMSE solution %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 if run_pa_lmmse&&(nargin==0)&&print_on,
  I = eye(nx); AA = NaN*ones(nz,nx); for i=1:nx, AA(:,i) = A.mult(I(:,i)); end;
  xhatLMMSE = xmean0 + xvar0.*(AA'*((AA*diag(xvar0)*AA'+diag(wvar))\(y-AA*xmean0)));
  mseLMMSE = 20*log10( norm(x-xhatLMMSE)/norm(x) );
  fprintf(1,'LMMSE: MSE = %5.1f dB\n', mseLMMSE);
 end;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % if desired, compute the phase-aware GAMP solution %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 if run_pa_gamp&&(nargin==0)&&print_on,
  opt_pa = GampOpt();
  opt_pa.stepWindow = 25;
  opt_pa.stepMax = 1;
  opt_pa.stepDecr = 0.5;
  opt_pa.stepIncr = 1.1;
  opt_pa.nit = 500;
  opt_pa.tol = 1e-6;	% iteration stopping tolerance [dflt:1e-6]
  outputEstPO = CAwgnEstimOut(y, max(wvar,wvar_min(end)*ones(nz,1)));
  xhatPO = gampEst(inputEst, outputEstPO, A, opt_pa);
  msePO = 20*log10( norm(x-xhatPO)/norm(x) );
  fprintf(1,'PO:    MSE = %5.1f dB\n', msePO);
 end;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % Run phase-retrieval GAMP, multiple times if needed %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 resGAMP = inf;
 cnt = 1;
 while resGAMP > threshTry,	% keep trying until output magnitudes look ok

  % initialize GAMP
  if (cnt==1)&&(exist('opt_pr')&&isfield(opt_pr,'xhat0')), % user-supplied
    xinit = opt_pr.xhat0;
    opt_gamp.xhat0 = xinit; 		
  else % random initialization
    if xreal,
      xinit = sparseRat*xmean0 + sqrt(sparseRat*xvar0).*randn(nx,1);
    else
      xinit = sparseRat*xmean0 + sqrt(sparseRat*xvar0/2).*(randn(nx,2)*[1;1i]);
    end;
    opt_gamp.xhat0 = init_gain*xinit; 		% notice init_gain!
  end;
  opt_gamp.xvar0 = xvar0 + abs(opt_gamp.xhat0-xmean0).^2;% note 2nd term!

  % run GAMP 
  [xhat,xvar,~,~,shat,svar,zhat,zvar,estHist] = gampEst(...
  					inputEst, outputEst, A, opt_gamp);
  resGAMP = 20*log10(norm(abs(A.mult(xhat))-abs_y)/norm(abs_y));% normalized residual
  if print_on,
    if nargin==0,
      mseGAMP = 20*log10(norm(x*sign(x'*xhat)-xhat)/norm(x));
      fprintf(1,'GAMP:  NMSE = %5.1f dB, NRES = %5.1f dB\n', [mseGAMP,resGAMP]);
    else
      fprintf(1,'GAMP:  NRES = %5.1f dB\n', resGAMP);
    end;
  end;

  % plot GAMP evolution
  if plot_gamp
    figure(plot_gamp); clf;
    nit = size(estHist.xhat,2);
    yhat = zeros(nz,nit); 
    for iter=1:nit, yhat(:,iter) = A.mult(estHist.xhat(:,iter)); end;
    resGAMP_ = 10*log10(sum((abs(yhat) ...
    			-abs_y*ones(1,nit)).^2,1)/norm(abs_y)^2);
    subplot(311)
     if nargin==0,
       mseGAMP_ = 10*log10(sum(abs(x*sign(x'*estHist.xhat) ...
    			-estHist.xhat).^2,1)/norm(x)^2);
       plot([mseGAMP_;resGAMP_].'); 
       legend('error','residual')
       ylabel('dB')
     else
       plot(resGAMP_); 
       ylabel('residual [dB]')
     end;
     title('GAMP')
     grid on;
    subplot(312)
     semilogy(-estHist.val)
     ylabel('cost')
     grid on;
    subplot(313)
     handy=plot(estHist.step,'.-');
     axe = axis; axis([axe(1:2),0.9*min(estHist.step),1.1*max(estHist.step)])
     set(handy,'MarkerSize',8)
     ylabel('step')
     xlabel('iter')
     grid on;
    drawnow
  end;%plot_gamp

  % if desired, run the gerchberg-saxton algorithm
  if run_gs&&(print_on|plot_gs),
    optGS.nit = 2000;             % maximum # iterations
    optGS.tol = 1e-7;             % stopping tolerance
    optGS.xreal = xreal;             % real or complex estimand?
    optGS.xhat0 = xinit;		% initialization
    [xhatGS,xhatGSHist] = gerchbergEst(abs_y,A,optGS);
    resGS = 20*log10(norm(abs(A.mult(xhatGS))-abs_y)/norm(abs_y));
    if print_on,
      if nargin==0,
        mseGS = 20*log10(norm(x*sign(x'*xhatGS)-xhatGS)/norm(x));
        fprintf(1,'GS:    NMSE = %5.1f dB, NRES = %5.1f dB\n', [mseGS,resGS]);
      else
        fprintf(1,'GS:    RES = %5.1f dB\n', resGS);
      end;
    end;

    % plot the GS evolution
    if plot_gs,
      figure(plot_gs); clf;
      nit = size(xhatGSHist,2);
      if nit==1, xhatGSHist = xhatGSHist*[1,1]; nit=2; end;
      yhatGSHist = zeros(nz,nit); 
      for iter=1:nit, yhatGSHist(:,iter) = A.mult(xhatGSHist(:,iter)); end;
      resGS_ = 10*log10(sum((abs(yhatGSHist) ...
      				-abs_y*ones(1,nit)).^2,1)/norm(y)^2);
      if nargin==0,
        mseGS_ = 10*log10(sum(abs(x*sign(x'*xhatGSHist) ...
      				-xhatGSHist).^2,1)/norm(x)^2);
        plot([mseGS_;resGS_].'); 
	legend('error','residual')
        ylabel('dB')
      else
        plot(resGS_); 
        ylabel('residual [dB]')
      end;
      xlabel('iter')
      title('GS')
      grid on
      drawnow
    end;% plot_gs
  end;% run_gs

  % plot signal estimates
  if plot_est,
    figure(plot_est); clf;
    handy = plot(abs(xhat));
    leg_str = strvcat('GAMP');
    hold on;
    if run_gs,
      handy = [handy; plot(abs(xhatGS),'g--')];
      leg_str = strvcat(leg_str,'GS');
    end;
    if nargin==0,
      if run_pa_lmmse,
        handy = [handy; plot(abs(xhatLMMSE),'k--')];
        leg_str = strvcat(leg_str,'PO-LMMSE');
      end;
      if run_pa_gamp,
        handy = [handy; plot(abs(xhatPO),'c--')];
        leg_str = strvcat(leg_str,'PO-GAMP');
      end;
      handy = [handy; plot(abs(x),'r--')];
      leg_str = strvcat(leg_str,'true');
    end;
    hold off;
    ylabel('absolute value')
    xlabel('signal index')
    legend(handy,leg_str);
    drawnow
  end;

  % break after too many GAMP re-initializations
  cnt = cnt+1;
  if cnt > maxTry,
    break;
  end;% if
 end;% while

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % run more GAMP iterations with less wvar limiting %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 estHist_ = estHist;		% initialize cumulative history
 for i=2:length(wvar_min),
   if 0 			% use predetermined schedule
     wvar_next = max(wvar,wvar_min(i)*ones(nz,1));
   else 			% use EM estimate
     if i==length(wvar_min),
       wvar_next = mean(zvar+2*(abs_y-abs(zhat)).^2)*ones(nz,1);	
     else
       wvar_next = mean(2*(abs_y-abs(zhat)).^2)*ones(nz,1);
     end;
     snrNext = 10*log10( mean(abs_y.^2)./mean(wvar_next) );
   end;
   outputEst = ncCAwgnEstimOut(abs_y,wvar_next); % notice variance limiting
   opt_gamp.xhat0 = xhat;	% warm start
   opt_gamp.xhatPrev0 = estHist.xhat(:,end-1);	% warm start
   opt_gamp.xvar0 = xvar;	% warm start
   opt_gamp.shat0 = shat;	% warm start
   opt_gamp.svar0 = svar;	% warm start
   opt_gamp.step = estHist.step(end);	% warm start
   opt_gamp.scaleFac = estHist.scaleFac(end);	% warm start
   %opt_gamp.tol = 1e-6;	% iteration stopping tolerance [dflt:1e-6]
   opt_gamp.tol = tol(i);	% iteration stopping tolerance
   [xhat,xvar,~,~,shat,svar,zhat,zvar,estHist] = gampEst(...
  					inputEst, outputEst, A, opt_gamp);
   resGAMPi = 20*log10(norm(abs(A.mult(xhat))-abs_y)/norm(abs_y));
   if print_on,
     if nargin==0,
       mseGAMPi = 20*log10(norm(x*sign(x'*xhat)-xhat)/norm(x));
       fprintf(1,'GAMPi: NMSE = %5.1f dB, NRES = %5.1f dB\n', [mseGAMPi,resGAMPi]);
     else
       fprintf(1,'GAMPi: NRES = %5.1f dB\n', resGAMPi);
     end;
   end;
   estHist_.xhat = [estHist_.xhat,estHist.xhat];
   estHist_.rhat = [estHist_.rhat,estHist.rhat];
   estHist_.rvar = [estHist_.rvar,estHist.rvar];
   estHist_.val =  [estHist_.val;estHist.val];
   estHist_.step = [estHist_.step;estHist.step];
   estHist_.pass = [estHist_.pass;estHist.pass];

   % plot GAMP evolution
   if plot_gamp,
     figure(plot_gamp); 
     nit = size(estHist_.xhat,2);
     yhat = zeros(nz,nit); for iter=1:nit, yhat(:,iter) = A.mult(estHist_.xhat(:,iter)); end;
     resGAMP_ = 10*log10(sum((abs(yhat) ...
    			-abs_y*ones(1,nit)).^2,1)/norm(abs_y)^2);
     subplot(311)
      if nargin==0,
        mseGAMP_ = 10*log10(sum(abs(x*sign(x'*estHist_.xhat) ...
    			-estHist_.xhat).^2,1)/norm(x)^2);
        plot([mseGAMP_;resGAMP_].'); 
        legend('signal','residual')
        ylabel('NMSE [dB]')
      else
        plot(resGAMP_); 
        ylabel('residual [dB]')
      end;
      title('GAMP')
      grid on;
     subplot(312)
      semilogy(-estHist_.val)
      ylabel('cost')
      grid on;
     subplot(313)
      handy=plot(estHist_.step,'.-');
      set(handy,'MarkerSize',8)
      axe = axis; axis([axe(1:2),0.9*min(estHist_.step),1.1*max(estHist_.step)])
      ylabel('step')
      xlabel('iter')
      grid on;
     drawnow
   end; % plot_gamp

   % plot signal estimates
   if plot_est,
     figure(plot_est); clf;
     handy = plot(abs(xhat));
     leg_str = strvcat('GAMP');
     if nargin==0,
       hold on;
       if run_pa_lmmse,
         handy = [handy; plot(abs(xhatLMMSE),'k--')];
         leg_str = strvcat(leg_str,'PO-LMMSE');
       end;
       if run_pa_gamp,
         handy = [handy; plot(abs(xhatPO),'c--')];
         leg_str = strvcat(leg_str,'PO-GAMP');
       end;
       if run_gs,
         handy = [handy; plot(abs(xhatGS),'g--')];
         leg_str = strvcat(leg_str,'GS');
       end;
       handy = [handy; plot(abs(x),'r--')];
       leg_str = strvcat(leg_str,'true');
       hold off;
       ylabel('absolute value')
       xlabel('signal index')
     end;
     legend(handy,leg_str);
     drawnow
   end; % plot_est

 end; % for i=2:length(wvar_min),

 %%%%%%%%%%%%%%%%%
 % write outputs %
 %%%%%%%%%%%%%%%%%
 out.xvar0 = mean(xvar0);
 out.wvar_min = wvar_min;
 out.threshTry = threshTry;
 out.tol = tol;
