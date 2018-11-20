% PBiGAMPsimple_BGAWGN is a wrapper for PBiGAMPsimple that assumes 
% Bernoulli-Gaussian signals, AWGN likelihood, and dense unstructured 
% measurement operators, and that makes multiple attempts from random 
% initializations if needed.  
%
% [results,estFin,estHist] = PBiGAMPiidBGawgn(y,A,Ac,Ab,optPB)
% where  
%   y: noisy measurements such that 
%           y(m) = b.'*A(m,:,:)*c + Ac(m,:)*b + Ab(m,:)*c + AWGN
%      where b,c are the unknown vectors of interest
%   A: 3D measurement array 
%   Ac,Ab: 2D measurement arrays (note: setting empty == setting to zero)
%   optPB: options structure (see below)
%   results: results structure (see below)
%   estFin: final estimates from PBiGAMP
%   estHist: estimate history from PBiGAMP
%
% optPB ... can be set at defaults using  "optPB = PBiGAMPiidBGawgnOpt();"
%  .EM: turn on EM-learning of b and c prior parameters [false]
%  .errTune: errfxn threshold to start tuning 
%  .errTry: errfxn threshold to stop trying
%  .maxTry: max number of random re-tries [10]
%  .maxIt: max number of PBiGAMP iterations per try [250]
%  .cmplx: complex valued BG prior and AWGN likelihood?
%  .verbose: verbose text output?
%  .plotFig: figure number for plot, or =0 for no plot
%  .sparB: assumed sparsity of b
%  .sparC: assumed sparsity of c
%  .meanB: assumed active mean of b
%  .meanC: assumed active mean of c
%  .varB: assumed active variance of b
%  .varC: assumed active variance of c
%  .wvar: assumed noise variance 
%  .errfxnB: implicit function to compute NMSE of b
%  .errfxnC: implicit function to compute NMSE of c
%  .stepInit: initial PBiGAMP damping parameter [0.0]
%  .stepIncr: rate at which to increase PBiGAMP damping parameter [1.05]
%  .varGainInit: multiplier on variance initialization [10]
%  .meanGainInit: multiplier on mean initialization [1]
%  .meanInitTypeB: in {'randn','spike'}
%  .meanInitTypeC: in {'randn','spike'}
%  .tol: stopping tolerance [1e-7]
%  .normTol: if norm(zhat)<normTol, then exit [1e-10]
%
% results 
%  .success: true if success was reached (in maxTry tries)
%  .numTry: total number of tries 
%  .err: lowest NMSE on Z of all tries
%  .errB: corresponding NMSE on B 
%  .errC: corresponding NMSE on C
%  .nit: corresponding number of iterations

function [results,estFin,estHist] = PBiGAMPiidBGawgn(y,A,Ac,Ab,optPB)

% compute important quantities
M = size(A,1);
Nb = size(A,2);
Nc = size(A,3);

% declare important parameters
errfxn = @(zhat) (norm(zhat-y,'fro')/norm(y,'fro'))^2; % normalized residual
wvar = optPB.wvar;              % assumed noise variance
SNRhat = mean(abs(y).^2)/wvar;  % estimated SNR
errTune = 2/SNRhat;             % NRSE threshold that triggers tuning

% declare input/output estimators 
if optPB.cmplx
  estInB = SparseScaEstim(CAwgnEstimIn(optPB.meanB,optPB.varB),optPB.sparB);
  estInC = SparseScaEstim(CAwgnEstimIn(optPB.meanC,optPB.varC),optPB.sparC);
  estOut = CAwgnEstimOut(y,wvar);
else
  estInB = SparseScaEstim(AwgnEstimIn(optPB.meanB,optPB.varB),optPB.sparB);
  estInC = SparseScaEstim(AwgnEstimIn(optPB.meanC,optPB.varC),optPB.sparC);
  estOut = AwgnEstimOut(y,wvar);
end

% handle autoTune stuff
if optPB.EM
    estInB.autoTune = true; % tune sparsity of B
    estInB.estim1.autoTune = true; % tune mean and variance of B
    estInC.autoTune = true; % tune sparsity of C
    estInC.estim1.autoTune = true; % tune mean and variance of C
    estOut.autoTune = true; % tune noise variance
    estOut.tuneMethod = 'EM';  % use Jeremy's EM-based noise-variance learning
    estOut.tuneDamp = 1;  % don't use autoTune damping
end

% set PBiGAMP options
opt = PBiGAMPOpt('adaptStep',false,'stepMin',0,'pvarStep',false,'zvarToPvarMax',inf,'uniformVariance',true);
if optPB.verbose||optPB.plotFig
  opt.error_functionB = optPB.errfxnB;
  opt.error_functionC = optPB.errfxnC;
end
%if optPB.EM, 
%  opr.saveState = true; % turn on state saving
%end
%opt.error_function = optPB.errfxnZ;
opt.error_function = errfxn; % used to control tuning and re-tries
opt.errTune = errTune; % start tuning at this level of error_function
opt.nit = optPB.maxIt; % max number of iterations
opt.step = optPB.stepInit; % initial damping factor [0.3]
opt.stepIncr = optPB.stepIncr; % increase damping slowly [1.006]
opt.stepMax = optPB.stepMax; % max damping factor [0.5]
opt.tol = optPB.tol; % stopping tolerance [1e-7]
opt.normTol = optPB.normTol; % stopping tolerance based on norm of z [1e-10]
opt.verbose = optPB.verbose; % verbose output?

% initialize PBiGAMP 
[mB,vB,~] = estInB.estimInit(); eB=abs(mB).^2+vB; % 2nd moment of B
[mC,vC,~] = estInC.estimInit(); eC=abs(mC).^2+vC; % 2nd moment of C
opt.bvar0 = optPB.varGainInit*eB; % initialize variance
opt.cvar0 = optPB.varGainInit*eC; % initialize variance
meanGainInitB = optPB.meanGainInitB; % mean multiplier for initialization [1]
meanGainInitC = optPB.meanGainInitC; % mean multiplier for initialization [1]

% run PBiGAMP with random re-starts
errTry = optPB.errTry;           % threshold to stop trying
errBest = inf;
for t=1:optPB.maxTry

    % initialize means
    switch lower(optPB.meanInitTypeB)
      case 'randn'
        if optPB.cmplx
          opt.bhat0 = sqrt(meanGainInitB*eB/2)*(randn(Nb,2)*[1;1i]); 
        else
          opt.bhat0 = sqrt(meanGainInitB*eB)*randn(Nb,1); 
        end
      case 'spike'
        opt.bhat0=zeros(Nb,1); opt.bhat0(randi(Nb))=sqrt(meanGainInitB*eB*Nb);
      otherwise
        error('unsupported meanInitTypeB') 
    end
    switch lower(optPB.meanInitTypeC)
      case 'randn'
        if optPB.cmplx
          opt.chat0 = sqrt(meanGainInitC*eC/2)*(randn(Nc,2)*[1;1i]); 
        else
          opt.chat0 = sqrt(meanGainInitC*eC)*randn(Nc,1); 
        end
      case 'spike'
        opt.chat0=zeros(Nc,1); opt.chat0(randi(Nc))=sqrt(meanGainInitC*eC*Nc);
      otherwise
        error('unsupported meanInitTypeC') 
    end

    % reset priors since previous attempts may have changed them
    estOut.wvar = wvar;  % reset prior
    estOut.disableTune = false;  % make sure not disabled 
    estInB.p1 = optPB.sparB; % reset prior
    estInB.estim1.mean0 = optPB.meanB; % reset prior
    estInB.estim1.var0 = optPB.varB; % reset prior
    estInB.disableTune = false;  % make sure not disabled 
    estInB.estim1.disableTune = false;  % make sure not disabled 
    estInC.p1 = optPB.sparC; % reset prior
    estInC.estim1.mean0 = optPB.meanC; % reset prior
    estInC.estim1.var0 = optPB.varC; % reset prior
    estInC.disableTune = false;  % make sure not disabled 
    estInC.estim1.disableTune = false;  % make sure not disabled 

    % run algorithm
    if optPB.plotFig||(nargout>2) % save history 
        [estFin,~,estHist] = PBiGAMPsimple(estInB,estInC,estOut,A,Ac,Ab,opt);
        errTune_ = errTune*ones(size(estHist.errZ));
        errTry_ = errTry*ones(size(estHist.errZ));
    else % don't save history (this is faster)
        estFin = PBiGAMPsimple(estInB,estInC,estOut,A,Ac,Ab,opt);
    end

    % plot results
    if optPB.plotFig
        figure(optPB.plotFig); clf
        plot(10*log10(estHist.errZ),'b-x')
        hold on
          plot(10*log10(estHist.errB),'r-x')
          plot(10*log10(estHist.errC),'g-x')
          plot(10*log10(errTry_),'r--')
          if optPB.EM
            plot(10*log10(errTune_),'b--')
          end
        hold off
        if optPB.EM
          legend('residual','b error','c error','try','tune')
        else
          legend('residual','b error','c error','try')
        end
        grid
        xlabel('iteration')
        ylabel('NMSE [dB]')
        tit_str = ['M=',num2str(M),', Nb=',num2str(Nb),', Nc=',num2str(Nc),...
            ', sparB=',num2str(optPB.sparB),', sparC=',num2str(optPB.sparC),...
            ', EM=',num2str(double(optPB.EM)),', try=',num2str(t)];
        title(tit_str)
        drawnow
    end

    err = errfxn(estFin.zhat);
    if err<errBest
      errBest = err;
      errBbest = optPB.errfxnB(estFin.bhat);
      errCbest = optPB.errfxnC(estFin.chat);
      nitBest = estFin.nit;
    end
    success = (err<errTry);
    if success, break; end % break on success
end

% fill results structure
results.success = success;
results.numTry = t;
results.nit = nitBest;
results.err = errBest;
results.errB = errBbest;
results.errC = errCbest;
if optPB.EM
  results.wvar = estOut.wvar;
  results.sparB = estInB.p1;
  results.sparC = estInC.p1;
  results.meanB = estInB.estim1.mean0;
  results.meanC = estInC.estim1.mean0;
  results.varB = estInB.estim1.var0;
  results.varC = estInC.estim1.var0;
end
