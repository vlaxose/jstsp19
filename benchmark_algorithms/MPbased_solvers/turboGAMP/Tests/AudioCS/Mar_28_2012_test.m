
[f,fs] = wavread('moz1_11kHz.wav');

%f = idct(gen_block(128,5,2,1,100,1))+randn(128,1)*0.01;
S = 50;
K = 3;
%  f = idct(gen_block(128,S,K,1,100,1));
Len = length(f);
N = 1500;
overlap = round((0/2)*N);   	% # of overlap samples in sliding window

fx = buffer(f, N, overlap);
m=floor(1/3*N);

% ****************************
% Configuration of EMturboGAMP
% ****************************
% % BERNOULLI-GAUSSIAN-MIXTURE OBJECT
% SigObj = GaussMix('sparsity_rate', cat(3, .05, .05, .05, .05), ...
%     'active_mean', cat(3, 0, 0, 0, 0), ...
%     'active_var', cat(3, 1e5, 1e4, 1e3, 1e2), ...
%     'learn_sparsity_rate', 'column', ...
%     'learn_active_mean', 'column', ...
%     'learn_active_var', 'column', ...
%     'init_params', 'true');
% % BERNOULLI-GAUSSIAN-MIXTURE OBJECT
% SigObj = GaussMix('sparsity_rate', cat(3, 0.05, 0.20), ...
%     'active_mean', cat(3, 0, 0), ...
%     'active_var', cat(3, 1e5, 1e4), ...
%     'learn_sparsity_rate', 'scalar', ...
%     'learn_active_mean', 'column', ...
%     'learn_active_var', 'column', ...
%     'init_params', 'true');
% BERNOULLI-GAUSSIAN OBJECT
SigObj = BernGauss('sparsity_rate', 0.10, ...
    'active_mean', 0, ...
    'active_var', 1e5, ...
    'learn_sparsity_rate', 'column', ...
    'learn_active_mean', 'column', ...
    'learn_active_var', 'column', ...
    'init_params', 'true');

% AWGN NOISE OBJECT
NoiseObj = GaussNoise('prior_var', 1e-8, ...
    'learn_prior_var', 'column', ...
    'init_params', 'true');
% NoiseObj = GaussMixNoise('PI', 0.10, ...
%         'NU0', 1e-8, ...
%         'NUratio', 1, ...
%         'learn_pi', 'false', ...
%         'learn_nu0', 'false', ...
%         'learn_nuratio', 'false');

% % (D+1)-ARY MARKOV CHAIN OBJECT
% SuppObj = MarkovChainD('p0d', cat(3, 0.25, 0.25, 0.25, 0.25), ...
%     'learn_p0d', 'true', ...
%     'dim', 'row');
% % BINARY MARKOV CHAIN OBJECT
% SuppObj = MarkovChain1('p01', 0.12, ...
%     'learn_p01', 'true', ...
%     'dim', 'row');
% % BINARY MARKOV RANDOM FIELD
% SuppObj = MarkovField('betaH', 0.3, ...
%  	'betaV', 0.3, ...
%   	'alpha', 0, ...
%  	'learn_beta', 'true', ...
%   	'learn_alpha', 'true', ...
%   	'maxIter', 20);
% NO SUPPORT STRUCTURE
SuppObj = SupportStruct();

% % GAUSS-MARKOV AMPLITUDE STRUCTURE
% AmplObj = GaussMarkov('alpha', 0.95, ...
%     'learn_alpha', 'false', ...
%     'dim', 'row', ...
%     'init_params', 'false');
% NO AMPLITUDE STRUCTURE
AmplObj = AmplitudeStruct();

% Runtime options
RunOpts = RunOptions('smooth_iters', 10, ...
    'verbose', true, 'tol', 5e-2);

% TURBOOPT OBJECT
TBobj = TurboOpt('Signal', SigObj, 'Noise', NoiseObj, 'SupportStruct', ...
    SuppObj, 'AmplitudeStruct', AmplObj, 'RunOptions', RunOpts);

% Print configuration to command window
TBobj.print();

% GAMP Options
GAMPopt = GampOpt();
GAMPopt.nit = 20;
GAMPopt.adaptStep = false;
% GAMPopt.bbStep = false;
GAMPopt.tol = 1e-3;
% GAMPopt.removeMean = false;

% *******************************
% Measurement matrix construction
% *******************************
randn('state',10000);
Phi = 1/sqrt(m)*randn(m,N);

A=zeros(m,N);
for k=1:m
A(k,:)=dct(Phi(k,:))'/N;
end
[w,h] = size(fx);

y2 = zeros(m,N);

% *******************
% Create measurements
% *******************
Y = Phi*fx;

% ************************
% Recover with EMturboGAMP
% ************************
tic
[Fhat, Fvar] = EMturboGAMP(Y, A, TBobj, GAMPopt);
toc
fhat = idct(Fhat/N);

% TNMSE = 10*log10(norm(fx - fhat, 'fro')^2 / norm(fx, 'fro')^2);
TNMSE = (1/h)*sum(sum(abs(fhat - fx).^2, 1) ./ sum(abs(fx).^2, 1), 2);
fprintf('EMturboGAMP TNMSE: %g dB\n', 10*log10(TNMSE))

% Plot results
clim = [min(min(10*log10(abs(dct(fx))))), max(max(10*log10(abs(dct(fx)))))];
figure(1); clf; colormap(colorspiral)
imagesc(10*log10(abs(dct(fx))), clim); colorbar
% imagesc(abs(dct(fx))); colorbar
xlabel('Signal block'); ylabel('|DCT[n]| (dB)')
title('Magnitude of DCT Coefficients of Audio Signal')
figure(2); clf; colormap(colorspiral)
imagesc(10*log10(abs(Fhat/N)), clim); colorbar
% imagesc((abs(Fhat))); colorbar
xlabel('Signal block'); ylabel('|DCT(F_{hat})| dB')
title(sprintf('Recovered Signal  |  TNMSE: %g dB', 10*log10(TNMSE)))

% sound(fhat(:), fs)

% for i = 1:1
%     y = Phi*fx(:,i);
% 
%     [ xhat_BG, ~, param ] = EMBGturboAMP(y, A, opt);
%     f_EMBGturboAMP(:,i)=idct(xhat_BG/N);
% 
%     % 
%     %    [ xhat_GM, ~, param ] = EMGMturboAMP(y, A);
%     %    f_EMGMturboAMP(:,i)=idct(xhat_GM/N);
% 
%     subplot(211)
%     plot(dct(fx(:,i))); hold on; plot(xhat_BG/N,'r'); hold off; ylabel('EMBGturboAMP')
%     %    subplot(212)
%     %    plot(dct(fx(:,i))); hold on; plot(xhat_GM/N,'r'); hold off; ylabel('EMGMturboAMP')
% 
%     disp(['NMSE_EMBGturboAMP:',num2str((norm(fx(:,i)-f_EMBGturboAMP(:,i))^2)/norm(fx(:,i))^2)]);
%     %    disp(['NMSE_EMGMturboAMP:',num2str((norm(fx(:,i)-f_EMGMturboAMP(:,i))^2)/norm(fx(:,i))^2)]);
% end
