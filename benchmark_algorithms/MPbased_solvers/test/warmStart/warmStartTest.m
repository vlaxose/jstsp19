% tests the transfer of de-meaned GAMP to meaned-GAMP

% Set path to the main directory
%addpath('../../main/');

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

 % Parameters
 nx = 200;         % Number of input components (dimension of x)
 nz = 100;         % Number of output components (dimension of y)
 sparseRat = 0.1;  % fraction of components of x that are non-zero
 snr = 60;         % SNR in dB.

 % Create a random sparse vector
 xmean0 = 0;        
 xvar0 = 1;
 x0 = normrnd(xmean0, sqrt(xvar0),nx,1); % a dense Gaussian vector
 x = x0.*(rand(nx,1) < sparseRat);       % insert zeros

 % Create a random measurement matrix
 A = (1/sqrt(nx))*randn(nz,nx);         % iid Gaussian, zero-mean
 %A = (1/sqrt(nx))*rand(nz,nx) - 0.4/sqrt(nx); % iid uniform, positive
 if 0,
   A = A-mean(A,2)*ones(1,nx);          % make exactly zero row-mean
   mu = rand(nz,1);                     % desired row-mean 
   A = A + mu*ones(1,nx);               % force row-mean
 elseif 0
   A = A-ones(nz,1)*mean(A,1);          % make exactly zero column-mean
   mu = 0.25*rand(1,nx);                % desired column-mean 
   A = A + ones(nz,1)*mu;               % force column-mean
 end;

 % Compute the noise level based on the specified SNR. 
 wvar = 10^(-0.1*snr)*(norm(A*x)^2)/nz;
 w = normrnd(0, sqrt(wvar), nz,1); % AWGN
 z = A*x;
 y = z + w;

 % Make multiple columns if desired
 if 0
   x = x*[1,1];
   z = z*[1,1];
   y = y*[1,1];
 end

 % Generate input estimation class
 % First, create an estimator for a Gaussian random variable (with no
 % sparsity)
 inputEst0 = AwgnEstimIn(xmean0, xvar0);
 
 % Then, create an input estimator from inputEst0 corresponding to a random
 % variable x that is zero with probability 1-sparseRat and has the
 % distribution of x in inputEst0 with probability sparseRat.
 inputEst = SparseScaEstim( inputEst0, sparseRat );
 
 % Output estimation class:  Use the 
 outputEst = AwgnEstimOut(y, wvar);

 % Set new GAMP options
 opt = GampOpt();
 opt.legacyOut = 0;
 opt.uniformVariance = 1; 
 opt.tol = 1e-8;
 opt.nit = 4;
 opt.adaptStep = 0;     
 opt.step = 0.1;
 opt.pvarStep = 1;      
 opt.stepMax = 1.0;     
 opt.stepIncr = 1.1;
 opt.stepMin = 0.1;
 opt.stepWindow = 0;
 opt.varNorm = 0;
 opt.removeMean = 0;
 %opt.step=2; opt.stepMax=2; % for testing adaptive stepsize
 opt.maxBadSteps = inf; opt.stepDecr = 0.5; % for testing automatic decrease of stepMax
 %opt.pvarMin=0; 

 % Set old GAMP options (should be same as above)
 opt0 = GampOpt_r395();
 opt0.legacyOut = 0;
 opt0.uniformVariance = 1; 
 opt0.tol = 1e-8;
 opt0.nit = 4;
 opt0.adaptStep = 0;    
 opt0.step = 0.1;       
 opt0.pvarStep = 1;     
 opt0.stepMax = 1.0;    
 opt0.stepIncr = 1.1;
 opt0.stepMin = 0.1;
 opt0.stepWindow = 0;
 opt0.varNorm = 0;      
 opt0.removeMean = 0;
 %opt0.step=2; opt0.stepMax=2; % for testing adaptive stepsize
 opt0.maxBadSteps = inf; opt0.stepDecr = 0.5; % for testing automatic decrease of stepMax
 %opt0.pvarMin=0; 

 format long;
 display(' ')
 %-----------------------------------------------------------

 display('Run new GAMP straight through:')
 opt.nit = 2*opt.nit;
 %opt.nit = 1003;
 [estFin0,optFin0,estHist0_] = gampEst(inputEst, outputEst, A, opt);
 opt.nit = opt.nit/2;
 figure(1000)
 gampShowHist(estHist0_,optFin0,x,z)
 nmseGAMP0= 20*log10( norm(x-estFin0.xhat)/norm(x) ) 

 %-----------------------------------------------------------

 display('Run new GAMP interrupted by a warm-start:')
 [estFin1,optFin1,estHist1_] = gampEst(inputEst, outputEst, A, opt);
 figure(1)
 gampShowHist(estHist1_,optFin1,x,z)

 % continue from warm start
 opt1_ = opt.warmStart(estFin1);
 %opt1_.nit = 1000;
 [estFin1,optFin1,estHist1] = gampEst(inputEst, outputEst, A, opt1_);
 estHist1_ = appendEstHist(estHist1_,estHist1);
 gampShowHist(estHist1_,optFin1,x,z)
 nmseGAMP1= 20*log10( norm(x-estFin1.xhat)/norm(x) ) 

 %-----------------------------------------------------------

 display('Run "no-adapt" version of GAMP interrupted by a warm-start:')
 [estFin2,optFin2,estHist2_] = gampEstNoAdapt(inputEst, outputEst, A, opt);
 figure(2)
 gampShowHist(estHist2_,optFin2,x,z)

 % continue from warm start
 opt2_ = opt.warmStart(estFin2);
 %opt2_.nit = 1000;
 [estFin2,optFin2,estHist2] = gampEstNoAdapt(inputEst, outputEst, A, opt2_);
 estHist2_ = appendEstHist(estHist2_,estHist2);
 gampShowHist(estHist2_,optFin2,x,z)
 nmseGAMP2= 20*log10( norm(x-estFin2.xhat)/norm(x) ) 

 %-----------------------------------------------------------

 display('Run "basic" version of GAMP straight through:')
 opt.nit = 2*opt.nit;
 %opt.nit = 1003;
 [estFin3,optFin3,estHist3_] = gampEstBasic(inputEst, outputEst, A, opt);
 opt.nit = opt.nit/2;
 figure(3)
 gampShowHist(estHist3_,optFin3,x,z)
 nmseGAMP3= 20*log10( norm(x-estFin3.xhat)/norm(x) ) 

 %-----------------------------------------------------------

 display('Run "basic" version of GAMP interrupted by a warm-start:')
 [estFin4,optFin4,estHist4_] = gampEstBasic(inputEst, outputEst, A, opt);
 figure(4)
 gampShowHist(estHist4_,optFin4,x,z)

 % continue from warm start
 opt4_ = opt.warmStart(estFin4);
 %opt4_.nit = 1000;
 [estFin4,optFin4,estHist4] = gampEstBasic(inputEst, outputEst, A, opt4_);
 estHist4_ = appendEstHist(estHist4_,estHist4);
 gampShowHist(estHist4_,optFin4,x,z)
 nmseGAMP4= 20*log10( norm(x-estFin4.xhat)/norm(x) ) 

 %-----------------------------------------------------------

 display('Run old GAMP straight through:')
 opt.nit = 2*opt.nit;
 %opt.nit = 1003;
 [estFin5,optFin5,estHist5_] = gampEst_r395(inputEst, outputEst, A, opt);
 opt.nit = opt.nit/2;
 figure(5)
 gampShowHist(estHist5_,optFin5,x,z)
 nmseGAMP5= 20*log10( norm(x-estFin5.xhat)/norm(x) ) 

 %-----------------------------------------------------------

 display('Run old GAMP interrupted by warm-start:')
 [estFin6,optFin6,estHist6_] = gampEst_r395(inputEst, outputEst, A, opt0);
 figure(6)
 gampShowHist(estHist6_,optFin6,x,z)

 % continue from warm start
 opt6_ = opt0.warmStartCopy(estFin6);
 %opt6_.nit = 1000;
 [estFin6,optFin6,estHist6] = gampEst_r395(inputEst, outputEst, A, opt6_);
 estHist6_ = appendEstHist(estHist6_,estHist6);
 gampShowHist(estHist6_,optFin6,x,z)
 nmseGAMP6= 20*log10( norm(x-estFin6.xhat)/norm(x) ) 

 %-----------------------------------------------------------

 display('Run new GAMP interrupted by a backwards-compatible warm-start:')
 [estFin7,optFin7,estHist7_] = gampEst(inputEst, outputEst, A, opt);
 figure(7)
 gampShowHist(estHist7_,optFin7,x,z)

 % Run new GAMP again from warm start
 opt7_ = opt.warmStartCopy(estFin7); % note use of "Old" method
 %opt7_.nit = 1000;
 [estFin7,optFin7,estHist7] = gampEst(inputEst, outputEst, A, opt7_);
 estHist7_ = appendEstHist(estHist7_,estHist7);
 gampShowHist(estHist7_,optFin7,x,z)
 nmseGAMP7= 20*log10( norm(x-estFin7.xhat)/norm(x) ) 

 %-----------------------------------------------------------
 format
 format long;

