% create a soft-thresholder EstimIn
debias = true;
EstIn0 = SoftThreshDMMEstimIn(1.5,'debias',debias);
%EstIn0 = SparseScaEstim(AwgnEstimIn(0,1),0.5);

% convert EstimIn to fxn handle 
fxnDenoise = @(rhat,rvar) EstIn0.estim(rhat,rvar);

% convert fxn handle to EstimIn (using Monte-Carlo divergence)
EstIn1 = FxnhandleEstimIn(fxnDenoise,'divMax',inf);

% setup simulation
rvar = 0.01; % try any value here
N = 10; % signal dimensionality
L = 200; % vary |rhat| across columns

% generate data and denoise
rhat = sign(randn(N,1))*linspace(-10*sqrt(rvar),10*sqrt(rvar),L);
[xhat0,xvar0] = EstIn0.estim(rhat,rvar);
[xhat1,xvar1] = EstIn1.estim(rhat,rvar);

% verify accuracy of divergence estimate
xvar_true_mean = mean(xvar0,1)
xvar_approx = xvar1(1,:)

% plot
subplot(221)
plot(rhat,xhat0,'+');
xlabel('rhat'); ylabel('xhat')
title('original')
grid on
subplot(223)
plot(rhat,xhat1,'+');
xlabel('rhat'); ylabel('xhat')
title('FxnhandleEstimIn')
grid on
subplot(222)
plot(rhat,xvar0,'+');
xlabel('rhat'); ylabel('xvar')
title('original')
grid on
subplot(224)
plot(rhat,xvar1,'+');
xlabel('rhat'); ylabel('xvar')
title('FxnhandleEstimIn')
grid on
