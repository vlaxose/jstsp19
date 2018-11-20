%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by EMNNAMP
%
% This function sets initializations for all unspecified NNGM and 
% exponential parameters to defaults.
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 8/01/13
% Change summary: 
%   v 1.0 (JV)- First release
%
% Version 1.0
%%
function stateFin = set_inits_EMNNAMP(Y, A, optALG, optEM)

[M,T] = size(Y);
[~,N] = size(A);

if ~isfield(optEM,'noise_var')
    optEM.SNRdB = min(100,optEM.SNRdB);	% prevent SNR > 100dB 
    if strcmp(optEM.outDim,'joint')
        optEM.noise_var = norm(Y,'fro')^2/T/(M*(1+10^(optEM.SNRdB/10)));
    else
        optEM.noise_var = sum(abs(Y).^2,1)/(M*(1+10^(optEM.SNRdB/10)));
    end
end

%Initialize output parameters
if ~optALG.laplace_noise
%Initialize noise variance if user has not done so already 
    
    if (optEM.noise_var == 0)
        warning('Since noise_var=0 can cause numerical problems, we have instead set it to a very small value')
        optEM.noise_var = norm(Y,'fro')^2/T/(M*1e10);
    elseif (optEM.noise_var < 0)
        error('noise_var<0 is not allowed')
    end
    
    %Define noise variance of measurements and pseudo measurements
    stateFin.noise_var = resize(optEM.noise_var,M,T);
else
    if ~isfield(optEM,'outLapRate')
        optEM.outLapRate = ones(M,T);
    end
    
    if (optEM.outLapRate < 0)
        error('outLapRate <0 is not allowed')
    end
    stateFin.outLapRate = resize(optEM.outLapRate,M,T);
end

if strcmp(optALG.alg_type,'NNGMAMP')
    if ~isfield(optEM,'L')
        optEM.L = 3;
    end
    L = optEM.L;

    %Determine undersampling ratio
    del = M/N;

    %Define cdf/pdf for Gaussian
    normal_cdf = @(x) 1/2*(1 + erf(x/sqrt(2)));

    %Define density of Gaussian
    normal_pdf = @(x) 1/sqrt(2*pi)*exp(-x.^2/2);

    alpha_grid = linspace(0,10,1024);
    rho_SE = (1 - (1/del)*((1+alpha_grid.^2).*normal_cdf(-alpha_grid)-alpha_grid.*normal_pdf(alpha_grid)))...
        ./(1 + alpha_grid.^2 - ((1+alpha_grid.^2).*normal_cdf(-alpha_grid)-alpha_grid.*normal_pdf(alpha_grid)));
    rho_SE = max(rho_SE);

    %Initialize tau
    if isfield(optEM,'tau')
        stateFin.tau = optEM.tau;
    else
        stateFin.tau = min(del*rho_SE,0.99);
    end;
    
    %load offline-computed initializations for GM parameters
    load('inits.mat','init')

    %Determine signal variance of matrix or of each column
    if strcmp(optEM.outDim,'col')
        sig_var = (sum(abs(Y).^2,1)-M*optEM.noise_var);
    else
        sig_var = (norm(Y,'fro')^2/T-M*optEM.noise_var);
    end;
    sig_var = resize(sig_var/sum(A.multSqTr(ones(M,1)))./stateFin.tau,N,T,L);

    if isfield(optEM,'active_weights')
        stateFin.active_weights = optEM.active_weights;
    else
        stateFin.active_weights = zeros(1,1,L);
        stateFin.active_weights(1,1,:) = init(L).active_weights;
        stateFin.active_weights = repmat(stateFin.active_weights, [N T 1]);
    end
    
    if isfield(optEM,'active_loc')
        stateFin.active_loc = optEM.active_loc;
    else
        stateFin.active_loc = zeros(1,1,L);
        %Shift to "positive" uniform prior
        stateFin.active_loc(1,1,:) = init(L).active_mean+0.5;
        stateFin.active_loc = repmat(stateFin.active_loc, [N T 1]).*...
            ((3*sig_var).^(1/2));
    end
    if isfield(optEM,'active_scales')
        stateFin.active_scales = optEM.active_scales;
    else
        stateFin.active_scales = zeros(1,1,L);
        stateFin.active_scales(1,1,:) = init(L).active_var;
        stateFin.active_scales = repmat(stateFin.active_scales, [N T 1])...
            .*(3*sig_var);
    end
        
    stateFin.tau = repmat(stateFin.tau,N,T);
elseif strcmp(optALG.alg_type,'NNLAMP')
    
    if ~isfield(optEM,'inExpRate')
        optEM.inExpRate = ones(N,T);
    end
    
    if (optEM.inExpRate < 0)
        error('inExpRate <0 is not allowed')
    end
    stateFin.inExpRate = resize(optEM.inExpRate,N,T);
    
end

return