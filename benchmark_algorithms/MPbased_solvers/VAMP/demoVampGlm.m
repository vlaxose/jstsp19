addpath('../stateEvo')

% handle random seed
if verLessThan('matlab','7.14')
  defaultStream = RandStream.getDefaultStream;
else
  defaultStream = RandStream.getGlobalStream;
end
if 1 % new RANDOM trial
  savedState = defaultStream.State;
  save random_state.mat savedState;
else % repeat last trial
  load random_state.mat
end
defaultStream.State = savedState;

% simulation parameters
L = 100; % # of measurement vectors
SNRdB = 40; % [40]
N = 512; % signal dimension [1024,512]
del = 4.0; % measurement rate M/N [0.5,4.0]
beta = 1/32; % sparsity rate K/N [0.1,1/32]
likeType = 'AWGN'; % in {'AWGN','Probit'}
svType = 'cond_num'; % in {'cond_num','spread','low_rank'}
cond_num = 1; % condition number
spread = 1; % amount to spread singular values (=1 means iid Gaussian A, =0 means frame)
low_rank = round(min(N,round(del*N))/8);
UType = 'Haar'; % in {'DFT','DCT','DHT','DHTrice','Haar'}
VType = 'Haar'; % in {'DFT','DCT','DHT','DHTrice','Haar'}
isCmplx = true; % simulate complex-valued case?

% algorithmic parameters
runGAMP = true; % run GAMP?
maxIt = 50; % max iterations for VAMP
tol = min(1e-3,max(1e-6,10^(-SNRdB/10))); % stopping tolerance for VAMP
damp = 0.9; % damping parameter
denoiser = 'BG'; % in {'BG','DMM','MAPLaplace'}
learnPrior = false; % automatically tune the denoiser?
learnLike = false; % automatically tune the likelihood?
learnGam1 = false; % automatically learn gam1?
init_genie = false;

% setup
xvar0 = 1; % prior variance of x elements
Afro2 = N; % squared Frobenius norm of matrix
M = round(del*N);
K = round(beta*N);
wvar = (Afro2/M)*10^(-SNRdB/10)*xvar0; % assumes norm(A,'fro')^2=N

% generate signal and noise
x = zeros(N,L); 
w = zeros(M,L); 
for l=1:L
  supp = randperm(N,K);
  if isCmplx
    x(supp,l) = sqrt(0.5*xvar0*N/K)*randn(K,2)*[1;1j];
    w(:,l) = sqrt(0.5*wvar)*randn(M,2)*[1;1j];
  else
    x(supp,l) = sqrt(xvar0*N/K)*randn(K,1);
    w(:,l) = sqrt(wvar)*randn(M,1);
  end
end

% generate linear transform
switch svType
  case 'spread', svParam = spread;
  case 'cond_num', svParam = cond_num;
  case 'low_rank', svParam = low_rank;
end
mat = genMatSVD(M,N,UType,svType,svParam,VType,'isCmplx',isCmplx,'Afro2',N);
A = mat.A;
U = mat.U;
V = mat.V;
d = mat.s.^2;

% crease noisy observations
z = A*x; 
SNRdB_test = 20*log10(norm(z)/norm(w));
switch likeType
  case 'AWGN'
    y = z + w;
  case 'Probit'
    if isCmplx
      error('Set isCmplx=false for Probit likelihood')
    else
      y = ((z+w)>0);
    end
end

% support-oracle performance bound for AWGN case
if strcmp(likeType,'AWGN')
  x0 = zeros(N,L);
  NMSEdB0 = nan(1,L);
  for l=1:L
    supp = find(x(:,l)~=0);
    A0 = A(:,supp);
    x0(supp,l) = (A0'*A0+(wvar/(xvar0*N/K))*eye(length(supp)))\(A0'*y(:,l));
    NMSEdB0(l) = 20*log10(norm(x0(:,l)-x(:,l))/norm(x(:,l)));
  end
end

% establish input denoiser
switch denoiser
case 'BG'
  spars = K/N;
  xvar1 = xvar0/spars;
  sparsInit = 1/N; 
  xvar0init = xvar0;
  xvar1init = xvar0init/sparsInit;
  tuneDim = 'joint';
  if learnPrior
    if isCmplx
      EstimIn = SparseScaEstim(CAwgnEstimIn(0,xvar1init,0,'autoTune',true,'mean0Tune',false,'tuneDim',tuneDim),sparsInit,0,'autoTune',true,'tuneDim',tuneDim);
    else
      EstimIn = SparseScaEstim(AwgnEstimIn(0,xvar1init,0,'autoTune',true,'mean0Tune',false,'tuneDim',tuneDim),sparsInit,0,'autoTune',true,'tuneDim',tuneDim);
    end
  else
    if isCmplx
      EstimIn = SparseScaEstim(CAwgnEstimIn(0,xvar1),spars);
    else
      EstimIn = SparseScaEstim(AwgnEstimIn(0,xvar1),spars);
    end
  end
case 'DMM'
  alpha = 1.5;
  debias = false;
  EstimIn = SoftThreshDMMEstimIn(alpha,'debias',debias);
  if learnPrior, 
    warning('learnPrior not implemented for SoftThreshDMM'); 
  end;
case 'MAPLaplace'
  lam = 1/sqrt(wvar);
  if learnPrior,
    EstimIn = SoftThreshEstimIn(lam,0,'autoTune',true,'counter',10) 
  else
    EstimIn = SoftThreshEstimIn(lam);
  end
otherwise
  error('unknown denoiser')
end

% establish likelihood
tuneDim = 'joint';
switch likeType
  case 'AWGN'
    wvarInit = 0.01 * norm(y,'fro')^2 / (M*L); % SNR ~= -20 dB
    if learnLike
      if isCmplx,
        EstimOut = CAwgnEstimOut(y,wvarInit,false,'autoTune',true,...
                        'tuneMethod','EM','tuneDamp',1,'tuneDim',tuneDim);
      else
        EstimOut = AwgnEstimOut(y,wvarInit,false,'autoTune',true,...
                        'tuneMethod','EM','tuneDamp',1,'tuneDim',tuneDim);
      end
    else
      if isCmplx,
        EstimOut = CAwgnEstimOut(y,wvar);
      else
        EstimOut = AwgnEstimOut(y,wvar);
      end
    end
  case 'Probit'
    wvarInit = 1e-10; % better choice?
    if learnLike, 
      EstimOut = ProbitEstimOut(y,0,wvarInit,false,'autoTune',true,...
                        'tuneMethod','EM','tuneDim',tuneDim);
    else
      EstimOut = ProbitEstimOut(y,0,wvar);
    end
end

% setup VAMP
vampOpt = VampGlmOpt;
vampOpt.nitMax = maxIt;
vampOpt.tol = tol;
vampOpt.damp = damp;
vampOpt.learnGam1 = learnGam1;
vampOpt.verbose = false;
vampOpt.fxnErr1 = @(x1,z1) 10*log10( sum(abs(x1-x).^2,1)./sum(abs(x).^2,1) );
vampOpt.fxnErr2 = @(x1,z1) 10*log10( sum(abs(...
        bsxfun(@times, x1, sum(conj(x1).*x,1)./sum(abs(x1).^2,1)) - x...
        ).^2,1)./sum(abs(x).^2,1) );
vampOpt.U = U;
vampOpt.V = V;
vampOpt.d = d;
if isCmplx, 
  vampOpt.r1init = eps*1i; % so that SparseScaEstim knows it's complex-valued
end

% run VAMP
[x1,vampEstFin] = VampGlmEst(EstimIn,EstimOut,A,vampOpt);
vampNMSEdB_ = vampEstFin.err1;
vampNMSEdB_debiased_ = vampEstFin.err2;
vampNit = vampEstFin.nit;

% run VAMP state evolution
estInAvg = EstimInAvg(EstimIn,x);
phat = sqrt(xvar0*Afro2/M)*randn(M,L);
switch likeType
  case 'AWGN'
    clear estOutAvg
    estOutAvg.mse = @(pvar) deal( 1/(1/wvar+1/pvar), 1/(1/wvar+1/pvar) ); 
    %estOutAvg = EstimOutAvg2(EstimOut,z); % monte-carlo given z
    %estOutAvg = EstimOutAvg(likeType,wvar,phat); % monte-carlo given phat
  case 'Probit'
    %estOutAvg = EstimOutAvg2(EstimOut,z); % monte-carlo given z
    estOutAvg = EstimOutAvg(likeType,wvar,phat); % monte-carlo given phat
end
vampSeNMSE = VampGlmSE(estInAvg,estOutAvg,d,N,M/N,vampNit)/xvar0;

% print learning
if learnPrior
  spars
  sparsEstimate = EstimIn.p1(1,:)
  xvar1
  xvar1estimate = EstimIn.estim1.var0(1,:)
end
if learnLike
  wvar
  switch likeType
    case 'AWGN'
      wvarEstimate = EstimOut.wvar(1) 
    case 'Probit'
      wvarEstimate = EstimOut.Var(1) 
  end
end

% setup and run GAMP
gampNit = 0;
if runGAMP
  Agamp = MatrixLinTrans(A);
  %optGAMP = GampOpt('legacyOut',false,'step',0.1,'stepIncr',1.05,'stepWindow',1,'uniformVariance',true,'tol',tol,'nit',500);
  optGAMP = GampOpt('legacyOut',false,'step',0.25,'stepMax',0.25,'adaptStep',false,'uniformVariance',true,'tol',tol,'nit',500);
  %optGAMP = GampOpt('legacyOut',false,'step',0.1,'stepMax',0.5,'adaptStep',false,'uniformVariance',true,'tol',tol,'nit',500);
  if init_genie
    optGAMP.xhat0 = xhat_genie;
    optGAMP.xvar0auto = true;
  end
  if learnPrior
    % reset these values
    EstimIn.p1 = sparsInit;
    EstimIn.estim1.var0 = xvar1init;
    EstimIn.estim1.mean0 = 0;
  end
  if learnLike
    % reset these values
    switch likeType
      case 'AWGN'
        EstimOut.wvar = wvarInit;
        EstimOut.tuneMethod = 'ML';
        EstimOut.tuneDim = 'col'; % seems to be important
      case 'Probit'
        EstimOut.Var = wvarInit;
        EstimOut.tuneMethod = 'ML';
        warning('NEED TO SET tuneDim=col')
    end
  end
  tstart = tic;
  [gampEstFin,optGampFin,gampEstHist] = gampEst(EstimIn,EstimOut,Agamp,optGAMP);
  time_gamp = toc(tstart);
  gampNit = gampEstFin.nit;
  gampNMSEdB_ = nan(L,gampNit);
  gampNMSEdB_debiased_ = nan(L,gampNit);
  for l=1:L
    xhat_ = gampEstHist.xhat((l-1)*N+[1:N],:);
    gampNMSEdB_(l,:) = 10*log10(sum(abs(xhat_-x(:,l)*ones(1,gampNit)).^2,1)/norm(x(:,l))^2);
    gain_ = conj(x(:,l)'*xhat_)./sum(abs(xhat_).^2,1);
    gampNMSEdB_debiased_(l,:) = 10*log10(sum(abs( bsxfun(@times,xhat_,gain_)-x(:,l)*ones(1,gampNit)).^2,1)/norm(x(:,l))^2);
  end
  %figure(3); clf; gampShowHist(gampEstHist,optGampFin,x); % debug GAMP
end

% plot results
figure(1); clf;
subplot(211) 
  % plot VAMP
  plot(1:vampNit-1,vampNMSEdB_(:,2:end),'.-') % first iteration is trivial
  leg_str = [];
  if (L>1)&&(L<6)
    for l=1:L
      leg_str = strvcat(leg_str,['VAMP, column ',num2str(l)]);
    end
  else
    leg_str = 'VAMP';
  end
  % plot GAMP
  if runGAMP
    hold on; 
      plot(1:gampNit-1,gampNMSEdB_(:,2:end),'.-')
    hold off; 
    if (L>1)&&(L<6)
      for l=1:L
        leg_str = strvcat(leg_str,['GAMP, column ',num2str(l)]);
      end
    else
      leg_str = strvcat(leg_str,'GAMP');
    end
  end
  % plot support oracle
  if strcmp(likeType,'AWGN')
    ax = gca; ax.ColorOrderIndex = 1; % use same colors
    hold on; 
      plot([1;max(vampNit,gampNit)],[1;1]*NMSEdB0,'--'); 
    hold off; 
    if (L>1)&&(L<6)
      for l=1:L
        leg_str = strvcat(leg_str,['support oracle, column ',num2str(l)]);
      end
    else
      leg_str = strvcat(leg_str,'support oracle');
    end
  end
  if L>1, legend(leg_str); else legend(leg_str,'Location','BestOutside'); end
  ylabel('NMSE [dB]')
  xlabel('iterations')
  grid on

subplot(212) 
  % plot VAMP
  plot(1:vampNit-1,vampNMSEdB_debiased_(:,2:end),'.-') % 1st VAMP iteration trivial
  leg_str = [];
  if (L>1)&&(L<6)
    for l=1:L
      leg_str = strvcat(leg_str,['VAMP, column ',num2str(l)]);
    end
  else
    leg_str = 'VAMP';
  end
  % plot GAMP
  if runGAMP
    hold on; 
      plot(1:gampNit-1,gampNMSEdB_debiased_(:,2:end),'.-')
    hold off; 
    if (L>1)&&(L<6)
      for l=1:L
        leg_str = strvcat(leg_str,['GAMP, column ',num2str(l)]);
      end
    else
      leg_str = strvcat(leg_str,'GAMP');
    end
  end
  % plot support oracle
  if strcmp(likeType,'AWGN')
    ax = gca; ax.ColorOrderIndex = 1; % use same colors
    hold on; 
      plot([1;max(vampNit,gampNit)],[1;1]*NMSEdB0,'--'); 
    hold off; 
    if (L>1)&&(L<6)
      for l=1:L
        leg_str = strvcat(leg_str,['oracle, column ',num2str(l)]);
      end
    else
      leg_str = strvcat(leg_str,'oracle');
    end
  end
  if L>1, legend(leg_str); else legend(leg_str,'Location','BestOutside'); end
  ylabel('debiased NMSE [dB]')
  xlabel('iterations')
  grid on

figure(2); clf;
  l = 1;
  if runGAMP, subplot(211); end;
    stem(x1(:,l))
    gain = sum(conj(x1(:,l)).*x(:,l),1)./sum(abs(x1(:,l)).^2);
    hold on; 
      stem(bsxfun(@times,x1(:,l),gain),'x'); 
      stem(x(:,l),'--'); 
    hold off;
    legend('VAMP','VAMP debiased','true')
    if L>1, title(['column ',num2str(l),' of ',num2str(L)]); end;
    xlabel('coefficient index')
    grid on;
  if runGAMP,
  subplot(212)
    xg = gampEstFin.xhat;
    stem(xg(:,l))
    gain = sum(conj(xg(:,l)).*x(:,l),1)./sum(abs(xg(:,l)).^2);
    hold on; 
      stem(bsxfun(@times,xg(:,l),gain),'x'); 
      stem(x(:,l),'--'); 
    hold off;
    legend('GAMP','GAMP debiased','true')
    xlabel('coefficient index')
    grid on;
  end

figure(3); clf;
  vampNMSE_avg = mean(10.^(vampNMSEdB_debiased_/10),1);
  gampNMSE_avg = mean(10.^(gampNMSEdB_debiased_/10),1);
  plot(1:vampNit,vampNMSE_avg,'.-',1:gampNit,gampNMSE_avg,'.-');
  set(gca,'YScale','log','XScale','log')
  hold on;
    semilogx(1:vampNit,vampSeNMSE,'k--');
  hold off;
  axis([1,gampNit,10^floor(log10(min([vampNMSE_avg,vampSeNMSE]))),1])
  legend('VAMP','GAMP','VAMP-SE')
  grid on
  xlabel('iteration')
  ylabel('debiased NMSE')
