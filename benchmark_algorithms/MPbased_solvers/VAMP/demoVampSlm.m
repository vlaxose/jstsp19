addpath('~/GAMPmatlab/stateEvo')

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
L = 1; % # of measurement vectors
SNRdB = 40; % [40]
N = 1024; % signal dimension [500]
del = 0.5; % measurement rate M/N [0.5]
rho = 0.2; % normalized sparsity rate E{K}/M [0.2]
svType = 'cond_num'; % in {'cond_num','spread','low_rank'}
cond_num = 1; % condition number [50 is limit for GAMP]
spread = 1; % amount to spread singular values (=1 means iid Gaussian A, =0 means frame) [6 is limit for GAMP]
low_rank = round(min(N,round(del*N))/2);
UType = 'Haar'; % in {'DFT','DCT','DHT','DHTrice','Haar'}
VType = 'Haar'; % in {'DFT','DCT','DHT','DHTrice','Haar'}
isCmplx = true; % simulate complex-valued case?
plot_traj = true; % plot trajectory of each column?
plot_error_stats = true; % plot error stats conditional on fixed A?
median_on = true; % use median instead of mean?
runEMBGAMP = true; % run GAMP?

% algorithmic parameters
maxit = 100; % max iterations for VAMP
tol = min(1e-3,max(1e-6,10^(-SNRdB/10))); % stopping tolerance for VAMP
damp = 0.95; % damping for VAMP
denoiser = 'BG'; % in {'BG','DMM','MAPLaplace'}
learnPrior = false; % automatically tune the denoiser?
learnNoisePrec = false; % automatically tune the noise variance?

% other defaults
fixed_K = true; % used fixed sparsity K=E{K}=round(rho*M)?
Afro2 = N; % squared Frobenius norm of matrix
xvar0 = 1; % prior variance of x coefs
xmean1 = 0; % prior mean of non-zero x coefs

% setup
M = round(del*N);
beta = rho*M/N; % probability of a non-zero coef
xvar1 = xvar0/beta; % prior variance of non-zero x coefs
wvar = (Afro2/M)*10^(-SNRdB/10)*beta*(xmean1^2+xvar1); 

% generate signal 
x = zeros(N,L);
for l=1:L
  if fixed_K
    supp = randperm(N,round(beta*N)); 
  else
    supp = find(rand(N,1)<beta); 
  end
  K = length(supp);
  %supp = 1:K; display('block support'); % for testing
  if isCmplx
    x(supp,l) = xmean1 + sqrt(0.5*xvar1)*randn(K,2)*[1;1j];
  else
    x(supp,l) = xmean1 + sqrt(xvar1)*randn(K,1);
  end
end
%x = abs(x); display('positive x')
%x =x(:,1)*ones(1,L); display('repeated x')

% generate noise 
if isCmplx
  w = sqrt(0.5*wvar)*(randn(M,L) + 1j*randn(M,L));
else
  w = sqrt(wvar)*randn(M,L);
end

% generate linear transform
switch svType
  case 'spread', svParam = spread;
  case 'cond_num', svParam = cond_num;
  case 'low_rank', svParam = low_rank;
end
shuffle = true;
randsign = true;
mat = genMatSVD(M,N,UType,svType,svParam,VType,...
                'isCmplx',isCmplx,'Afro2',N,...
                'shuffle',shuffle,'randsign',randsign);
U = mat.U;
s = mat.s;
V = mat.V;
if plot_error_stats && strcmp(UType,'Haar') && (M>N)
  if isCmplx
    A = (randn(M,N)+1i*randn(M,N))/sqrt(2*M);
  else
    A = randn(M,N)/sqrt(M);
  end
  [U,~,~] = svd(A);
end
if plot_error_stats && strcmp(VType,'Haar') && (M<N)
  % do manually because genMatSVD effectively uses an "economy" svd
  if isCmplx
    A = (randn(M,N)+1i*randn(M,N))/sqrt(2*M);
  else
    A = randn(M,N)/sqrt(M);
  end
  [~,~,V] = svd(A);
end
A = U*spdiags(s(1:min(M,N)),0,M,N)*V';
d = [s.^2;zeros(M-length(s),1)]; % need length(d)=M

% generate observation
z = A*x; 
SNRdB_test = 20*log10(norm(z(:))/norm(w(:)));
y = z + w;

% support-oracle performance bound
x0 = zeros(N,L);
oracleNMSEdB = nan(L,1);
for l=1:L
  supp = find(x(:,l)~=0);
  A0 = A(:,supp);
  x0(supp,l) = (A0'*A0+(wvar/xvar1)*eye(length(supp)))\(A0'*y(:,l)); 
  oracleNMSEdB(l) = 20*log10(norm(x0(:,l)-x(:,l))/norm(x(:,l)));
end

% establish denoiser
switch denoiser
case 'BG'
  if learnPrior
    if isCmplx
      if beta<1
        EstimIn = SparseScaEstim(CAwgnEstimIn(0,10*xvar1,0,'autoTune',true,'tuneDim','col'),0.1*beta,0,'autoTune',true,'tuneDim','col');
      elseif beta==1
        EstimIn = CAwgnEstimIn(0,10*xvar1,0,'autoTune',true,'tuneDim','col');
      else
        error('invalid rho since rho>N/M')
      end
    else
      if beta<1
        EstimIn = SparseScaEstim(AwgnEstimIn(0,10*xvar1,0,'autoTune',true,'tuneDim','col'),0.1*beta,0,'autoTune',true,'tuneDim','col');
      elseif beta==1,
        EstimIn = AwgnEstimIn(0,10*xvar1,0,'autoTune',true,'tuneDim','col');
      else
        error('invalid rho since rho>N/M')
      end
    end
  else
    if isCmplx
      EstimIn = SparseScaEstim(CAwgnEstimIn(0,xvar1),beta);
    else
      EstimIn = SparseScaEstim(AwgnEstimIn(0,xvar1),beta);
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

% setup VAMP
vampOpt = VampSlmOpt;
vampOpt.nitMax = maxit;
vampOpt.tol = tol;
vampOpt.damp = damp;
vampOpt.learnNoisePrec = learnNoisePrec;
if learnNoisePrec
  vampOpt.learnNoisePrec = true;
else
  vampOpt.learnNoisePrec = false;
  vampOpt.NoisePrecInit = 1/wvar; 
end
vampOpt.learnNoisePrecAlg = 'EM';
vampOpt.learnNoisePrecNit = 100;
vampOpt.learnNoisePrecTol = 0.01;
vampOpt.verbose = false;
vampOpt.fxnErr = @(x2) 10*log10( sum(abs(x2-x).^2,1)./sum(abs(x).^2,1) ); 
vampOpt.U = U;
vampOpt.d = d;

% run VAMP
if plot_error_stats
  [~,vampEstFin,vampEstHist] = VampSlmEst(EstimIn,y,A,vampOpt);
else
  [~,vampEstFin] = VampSlmEst(EstimIn,y,A,vampOpt);
end
vampNMSEdB_ = vampEstFin.err; 
vampNit = vampEstFin.nit;

% setup and run EM-BG-GAMP
gampNit = 0;
if runEMBGAMP
  Agamp = MatrixLinTrans(A);
  clear optEM optGAMP;
  optEM.heavy_tailed = false;
  optEM.robust_gamp = true;
  optGAMP.removeMean = false;
  tstart = tic;
  [~,EMfin,gampEstHist,~,optGAMPfin] = EMBGAMP(y,Agamp,optEM,optGAMP);
  time_gamp = toc(tstart);
  gampNit = length(gampEstHist.it);
  gampNMSEdB_ = nan(L,gampNit);
  for l=1:L
    gampNMSEdB_(l,:) = 10*log10(sum(abs(gampEstHist.xhat((l-1)*N+[1:N],:)-x(:,l)*ones(1,gampNit)).^2,1)/norm(x(:,l))^2);
  end
  %figure(2); clf; gampShowHist(gampEstHist,optGAMPfin,x); % debug GAMP
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plot_traj
  % plot VAMP
  figure(1); clf;
  semilogx(1:vampNit,vampNMSEdB_,'.-')
  leg_str = [];
  if L>1
    for l=1:L
      leg_str = strvcat(leg_str,['VAMP, column ',num2str(l)]);
    end
  else
    leg_str = 'VAMP';
  end
  % plot GAMP
  if runEMBGAMP
    hold on;
      semilogx(1:gampNit,gampNMSEdB_,'.--')
    hold off;
    if L>1
      for l=1:L
        leg_str = strvcat(leg_str,['EMBGAMP, column ',num2str(l)]);
      end
    else
      leg_str = strvcat(leg_str,'EMBGAMP');
    end
  end
  % plot support oracle
  ax = gca; ax.ColorOrderIndex = 1; % use same colors
  hold on; 
    semilogx([1;max(vampNit,gampNit)],oracleNMSEdB*[1,1],'-.'); 
  hold off;
  if L>1
    for l=1:L
      leg_str = strvcat(leg_str,['oracle, column ',num2str(l)]);
    end
  else
    leg_str = strvcat(leg_str,'oracle');
  end
  % legend
  if L<=5 
    legend(leg_str); 
  elseif L<=10
    legend(leg_str,'Location','BestOutside'); 
  end
  if median_on
    ylabel('median NMSE [dB]')
  else
    ylabel('average NMSE [dB]')
  end
  xlabel('iterations')
  grid on
  axis([1,max(vampNit,gampNit),5*floor(min([vampNMSEdB_(:);oracleNMSEdB])/5),1])
end % plot_traj

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run state evolution
estInAvg = EstimInAvg(EstimIn,x);
gam1init = 1/(xvar0 + wvar*sum(d)/sum(d.^2)); % VampSlm default
%gam1init = 1000 % to test replica, try starting from near-perfect initialization
vampSeNMSE = VampSlmSE(estInAvg,d,N,wvar,vampNit,gam1init,damp)./xvar0;

% plot state evolution
figure(2); clf;
if median_on
  vampNMSE_avg = median(10.^(vampNMSEdB_/10),1);
  oracleNMSE_avg = median(10.^(oracleNMSEdB/10),1);
  if runEMBGAMP, gampNMSE_avg = median(10.^(gampNMSEdB_/10),1); end
else
  vampNMSE_avg = mean(10.^(vampNMSEdB_/10),1);
  oracleNMSE_avg = mean(10.^(oracleNMSEdB/10),1);
  if runEMBGAMP, gampNMSE_avg = mean(10.^(gampNMSEdB_/10),1); end
end
plot(1:vampNit,vampNMSE_avg,'+-');
set(gca,'YScale','log','XScale','log')
hold on;
  semilogx(1:vampNit,vampSeNMSE,'o-'); 
  if runEMBGAMP 
    semilogx(1:gampNit,gampNMSE_avg,'x-'); 
    semilogx([1,gampNit],[1,1]*oracleNMSE_avg,'-.');
  else
    semilogx([1,vampNit],[1,1]*oracleNMSE_avg,'-.');
  end
hold off;
if runEMBGAMP
  legend('VAMP','VAMP SE','EMBGAMP','oracle')
  axis([1,gampNit,10^floor(log10(min([vampNMSE_avg,oracleNMSE_avg,vampSeNMSE]))),1])
else
  legend('VAMP','VAMP SE','oracle')
  axis([1,vampNit,10^floor(log10(min([vampNMSE_avg,oracleNMSE_avg,vampSeNMSE]))),1])
end
grid on
xlabel('iteration')
if median_on
  ylabel('median NMSE [dB]')
else
  ylabel('average NMSE [dB]')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot VAMP error stats
if plot_error_stats
  if damp~=1, warning('with damping~=1, error stats may be off'); end;

  gam1_ = vampEstHist.gam1;
  gam2_ = vampEstHist.gam2;
  eta1_ = vampEstHist.eta1;
  eta2_ = vampEstHist.eta2;
  xvar2_ = vampEstHist.xvar2;
  alf_ = gam2_./eta1_;

  % errors
  e1_ = bsxfun(@minus,vampEstHist.r1,x);
  e2_ = bsxfun(@minus,vampEstHist.r2,x);
  q1_ = bsxfun(@minus,vampEstHist.x1,x);
  q2_ = bsxfun(@minus,vampEstHist.x2,x);

  % rotated errors
  e1tilde_ = zeros(N,L,vampNit);
  e2tilde_ = zeros(N,L,vampNit);
  q1tilde_ = zeros(N,L,vampNit);
  q2tilde_ = zeros(N,L,vampNit);
  for i=1:vampNit 
    e1tilde_(:,:,i) = V'*e1_(:,:,i);
    e2tilde_(:,:,i) = V'*e2_(:,:,i);
    q1tilde_(:,:,i) = V'*q1_(:,:,i);
    q2tilde_(:,:,i) = V'*q2_(:,:,i);
  end

  % find bad indices
  q2med = median(mean(abs(q2_(:,:,end)).^2,1),2); % median at final iteration
  good = find(mean(abs(q2_(:,:,end)).^2,1) < 10*q2med); % good realizations
  if length(good)<L
    warning(['throwing out %i bad realizations'],L-length(good))
  end

  % avg across realizations
  e1ms_ = squeeze(mean(abs(e1_(:,good,:)).^2,2)); 
  e2ms_ = squeeze(mean(abs(e2_(:,good,:)).^2,2)); 
  q1ms_ = squeeze(mean(abs(q1_(:,good,:)).^2,2)); 
  q2ms_ = squeeze(mean(abs(q2_(:,good,:)).^2,2)); 
  e1tildems_ = squeeze(mean(abs(e1tilde_(:,good,:)).^2,2)); 
  e2tildems_ = squeeze(mean(abs(e2tilde_(:,good,:)).^2,2)); 
  q1tildems_ = squeeze(mean(abs(q1tilde_(:,good,:)).^2,2)); 
  q2tildems_ = squeeze(mean(abs(q2tilde_(:,good,:)).^2,2)); 

  % rotated error stats
  i = 5;  % iteration to plot at (note: need i>1!)
  gamw = vampEstHist.gamwHat(i);
  dd = [d;zeros(N-M,1)];
  sigmai = (gamw*dd*(1./alf_(good,i))')./bsxfun(@plus,gamw*dd,gam1_(good,i).'); % N by L
  e1covi = ones(N,1)*mean(1./(gam1_(good,i).'),2); 
  e1cov_avg = mean(1./gam1_,1);
  q1covi = mean(1./bsxfun(@plus,gamw*dd,gam1_(good,i).'),2); 
  q1cov_avg = mean(1./eta1_,1);
  e2covi = mean( bsxfun(@ldivide, gam1_(good,i).', 1-bsxfun(@times,2-1./alf_(good,i)', sigmai ) ) ,2);
  e2cov_avg = mean(1./gam2_,1);
  %q2covi = mean(xvar2_(:,good,i),2); 
  q2covi = ones(N,1)*mean(1./eta2_(good,i),1); 
  q2cov_avg = mean(1./eta2_,1);

  % plot avg rotated errors versus index 
  figure(3); clf;
   semilogy(e1ms_(:,i))
   hold on;
     semilogy(e1covi,'--')
     semilogy(q1tildems_(:,i))
     semilogy(q1covi,'--')
     semilogy(e2tildems_(:,i))
     semilogy(e2covi,'--')
     semilogy(q2ms_(:,i))
     semilogy(q2covi,'--')
   hold off;
   legend('avg{ |r1-x|.^2 }', 'theory',...
          'avg{ |V''(x1-x)|.^2 }', 'theory',...
          'avg{ |V''(r2-x)|.^2 }', 'theory',...
          'avg{ |x2-x|.^2 }', 'theory',...
          'Location','Best')
   title(['iteration=',num2str(i)])
   xlabel('coef index');
   grid on

  % plot avg rotated errors versus iteration 
  figure(4); clf;
   loglog(mean(e1ms_(:,:)),'.-')
   hold on;
     loglog(e1cov_avg,'--')
     loglog(mean(q1ms_(:,:)),'.-')
     loglog(q1cov_avg,'--')
     loglog(mean(e2ms_(:,:)),'.-')
     loglog(e2cov_avg,'--')
     loglog(mean(q2ms_(:,:)),'.-')
     loglog(q2cov_avg,'--')
   hold off;
   legend('avg{ |r1-x|.^2 }', '1/gam1',...
          'avg{ |x1-x|.^2 }', '1/eta1',...
          'avg{ |r2-x|.^2 }', '1/gam2',...
          'avg{ |x2-x|.^2 }', '1/eta2',...
          'Location','Best')
   xlabel('iteration');
   grid on

  % plot each avg rotated error versus iteration and coefficients
  figure(5); clf;
   waterfall(10*log10(e1ms_))
   hold on; plot3(i*ones(N,1),1:N,10*log10(e1covi),'o'); hold off;
   hold on; plot3(1:vampNit,ones(vampNit,1),10*log10(e1cov_avg),'s'); hold off;
   legend('simulation','theory','1/gamma1','Location','Best')
   %title('avg{ |V''(r1-x)|.^2 }: error at input to linear stage','Interpreter','none');
   title('avg{ |r1-x|.^2 }: error at input to linear stage','Interpreter','none');
   ylabel('coef index');
   xlabel('iteration')
   view(40,30)
  figure(6); clf;
   waterfall(10*log10(q1tildems_))
   hold on; plot3(i*ones(N,1),1:N,10*log10(q1covi),'o'); hold off;
   hold on; plot3(1:vampNit,ones(vampNit,1),10*log10(q1cov_avg),'s'); hold off;
   legend('simulation','theory','1/eta1','Location','Best')
   title('avg{ |V''(x1-x)|.^2 }: error at output of linear stage','Interpreter','none');
   ylabel('coef index');
   xlabel('iteration')
   view(40,30)
  figure(7); clf;
   waterfall(10*log10(e2tildems_))
   hold on; plot3(i*ones(N,1),1:N,10*log10(e2covi),'o'); hold off;
   hold on; plot3(1:vampNit,ones(vampNit,1),10*log10(e2cov_avg),'s'); hold off;
   legend('simulation','theory','1/gamma2','Location','Best')
   title('avg{ |V''(r2-x)|.^2 }: error at input to nonlinear stage','Interpreter','none');
   ylabel('coef index');
   xlabel('iteration')
   view(40,30)
  figure(8); clf;
   waterfall(10*log10(q2ms_))
   hold on; plot3(i*ones(N,1),1:N,10*log10(q2covi),'o'); hold off;
   hold on; plot3(1:vampNit,ones(vampNit,1),10*log10(q2cov_avg),'s'); hold off;
   title('avg{ |x2-x|.^2 }: error at output of nonlinear stage','Interpreter','none');
   legend('simulation','theory','1/eta2','Location','Best')
   ylabel('coef index');
   xlabel('iteration')
   view(40,30)

  Re2tilde = (1/L)*squeeze(e2tilde_(:,:,i))*squeeze(e2tilde_(:,:,i))';
  figure(9); clf;
   %imagesc(10*log10(abs(Re2tilde))); colorbar
   %mesh(10*log10(abs(Re2tilde)));
   mesh(abs(Re2tilde));
   view(-28,12)
   title(['autocorr{ |V''(r2-x)|.^2 } at iteration ',num2str(i)],'Interpreter','none');
   xlabel('coef index');
   ylabel('coef index');

  % check some fixed-point stuff
  nmse = @(a,b) 20*log10(norm(a-b)/norm(b));
  i=vampNit; l=1;
  e1 = e1_(:,l,i);
  e2 = e2_(:,l,i);
  q1 = q1_(:,l,i);
  q2 = q2_(:,l,i);
  e1tilde = e1tilde_(:,l,i);
  e2tilde = e2tilde_(:,l,i);
  q1tilde = q1tilde_(:,l,i);
  q2tilde = q2tilde_(:,l,i);
  gam1 = gam1_(l,i);
  gam2 = gam2_(l,i);
  eta1 = eta1_(l,i);
  %nmse( gam1*e1 + gam2*e2 , eta1*q1 )
   
end % plot_error_stats
