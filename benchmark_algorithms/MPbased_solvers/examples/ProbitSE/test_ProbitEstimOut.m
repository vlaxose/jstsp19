M = 1e7; % # observations
tauw = 0.01; % observation noise variance
taup = 1; % prior variance on pre-quantized outputs

% generate realizations 
%phat = linspace(-1,1,M).'; % prior mean (i.e., noisy estimates) of pre-quantized outputs
phat = abs(randn(M,1)); % prior mean (i.e., noisy estimates) of pre-quantized outputs 
v = sqrt(taup)*randn(M,1); 
z = phat + v; % pre-quantized outputs 
w = sqrt(tauw)*randn(M,1); % AWGN measurement noise
y = ((z+w)>0); % 1-bit quantization (i.e., probit measurement channel)

% estimation under mismatch
mismatch_ = logspace(-0.1,0.1,5) % mismatch parameter
tauz_ = nan(size(mismatch_));
zerrmse_ = nan(size(mismatch_));
tauz_avg_ = nan(size(mismatch_));
for i=1:length(mismatch_)

  mismatch = mismatch_(i);

  if 1 % ideal case: {phat,taup} are correct
    phat_assumed = phat;
    taup_assumed = taup;
  else % this happens at first GAMP iteration (since xhat=0 => phat=0)
    phat_assumed = zeros(M,1); 
    taup_assumed = mean(phat.^2) + taup;
  end

  EstimOut = ProbitEstimOut(y,0,tauw);
  [zhat,tauz] = EstimOut.estim(phat_assumed,mismatch*taup_assumed*ones(M,1));
  zerr = zhat-z; % estimation error
  zerrmse_(i) = mean(zerr.^2); % MSE
  tauz_avg_(i) = mean(tauz); % predicted MSE

  figure(1); clf;
  hist(zerr,100)
  ylabel('histogram')
  xlabel('estimation error')
  grid on
  drawnow

end

figure(2); clf;
plot(mismatch_,zerrmse_,'o-', mismatch_,tauz_avg_,'x-')
legend('mse','predicted mse')
xlabel('scaling parameter (to test mismatch)')
grid on
