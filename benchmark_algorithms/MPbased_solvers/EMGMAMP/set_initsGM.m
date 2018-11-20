%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by EMGMAMP.
%
% This function sets initializations for all unspecified GM parameters to
% defaults.
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 6/05/13
% Change summary: 
%   v 1.0 (JV)- First release
%   v 1.1 (JV)- Handle noise variance set to 0, fixed use of wrong norm 
%   v 1.2 (JV)- Fixed noise_dim and sig_dim problems and cleaned up
%               variable names
%
% Version 1.2

function [lambda, omega, theta, phi, optEM] = set_initsGM(optEM, Y, A, M, N, T)

%Initialize noise variance if user has not done so already 
if ~isfield(optEM,'noise_var')
    optEM.SNRdB = min(100,optEM.SNRdB);	% prevent SNR > 100dB 
    if strcmp(optEM.noise_dim,'col')
        optEM.noise_var = sum(abs(Y).^2,1)/(M*(1+10^(optEM.SNRdB/10)));
    else
        optEM.noise_var = norm(Y,'fro')^2/T/(M*(1+10^(optEM.SNRdB/10)));
    end
end
if (optEM.noise_var == 0)
    warning('Since noise_var=0 can cause numerical problems, we have instead set it to a very small value')
    optEM.noise_var = norm(Y,'fro')^2/T/(M*1e10);
elseif (optEM.noise_var < 0)
    error('noise_var<0 is not allowed')
end

%If not defined by user, see if input is complex 
if ~isfield(optEM,'cmplx_in')
    if ~isreal(A.multTr(randn(M,1))) || ~isreal(Y)
        optEM.cmplx_in = true;
    else
        optEM.cmplx_in = false;
    end
end

%If not defined by user, see if output is complex 
if ~isfield(optEM,'cmplx_out')
    if ~isreal(Y)
        optEM.cmplx_out = true;
    else
        optEM.cmplx_out = false;
    end
end
if ~optEM.cmplx_out && ~isreal(Y)
    error('Since measurements are complex, must set optEM.cmplx_out=true')
end

%Determine undersampling ratio
del = M/N;

%Initialize all parameters
if ~isfield(optEM,'active_weights') 
    L = optEM.L;
else
    L = size(optEM.active_weights,1);
end

%Define cdf/pdf for Gaussian
normal_cdf = @(x) 1/2*(1 + erf(x/sqrt(2)));

%Define density of Gaussian
normal_pdf = @(x) 1/sqrt(2*pi)*exp(-x.^2/2);

alpha_grid = linspace(0,10,1024);
rho_SE = (1 - (2/del)*((1+alpha_grid.^2).*normal_cdf(-alpha_grid)...
    -alpha_grid.*normal_pdf(alpha_grid)))...
    ./(1 + alpha_grid.^2 - 2*((1+alpha_grid.^2).*normal_cdf(-alpha_grid)...
    -alpha_grid.*normal_pdf(alpha_grid)));
rho_SE = max(rho_SE);

%Initialize lambda
if isfield(optEM,'lambda')
    lambda = optEM.lambda;
else
    lambda = min(del*rho_SE,0.999);
end;

%load offline-computed initializations for GM parameters
load('inits.mat','init')

%Determine signal variance of matrix or of each column
if strcmp(optEM.noise_dim,'col')
    sig_var = (sum(abs(Y).^2,1)-M*optEM.noise_var);
else
    sig_var = (norm(Y,'fro')^2/T-M*optEM.noise_var);
end;
sig_var = resize(sig_var/sum(A.multSqTr(ones(M,1)))./lambda,N,T,L);

% Initialize Gaussian Mixture parameters
if ~optEM.heavy_tailed
    scale_factor = zeros(1,1,L);
    omega = zeros(1,1,L);
    
    %initialize active weights with pre-defined inputs or defaults
    if isfield(optEM,'active_weights')
        if (size(optEM.active_weights,2) > 1)
            omega = zeros(1,T,L);
            omega(1,:,:) = optEM.active_weights';
        else
            omega(1,1,:) = optEM.active_weights;
        end
    else
        omega(1,1,:) = init(L).active_weights;
    end;
    
    %initialize active variances with pre-defined inputs or defaults
    if isfield(optEM,'active_var')
        if (size(optEM.active_var,2) > 1)
            phi = zeros(1,T,L);
            phi(1,:,:) = optEM.active_var';
        else
            phi(1,1,:) =  optEM.active_var;
        end
    else
        scale_factor(1,1,:) = init(L).active_var;
        phi =  repmat(scale_factor,[N T 1])*12.*sig_var;
    end;

    %initialize active means with pre-defines inputs or defaults
    if isfield(optEM,'active_mean')
         if (size(optEM.active_mean,2) > 1)
            theta = zeros(1,T,L);
            theta(1,:,:) = optEM.active_mean.';
         else
            theta(1,1,:) = optEM.active_mean;
         end
    else
        scale_factor(1,1,:) = init(L).active_mean;
        theta = repmat(scale_factor, [N T 1]).*sqrt(12*sig_var);
    end;  

%Define Heavy tailed initializations.  Override some user-defined inputs
else
    %Initialize weights with pre-defined inputs or defaults
    if isfield(optEM,'active_weights')
        if (size(optEM.active_weights,2) > 1)
            omega = zeros(1,T,L);
            omega(1,:,:) = optEM.active_weights';
        else
            omega(1,1,:) = optEM.active_weights;
        end
    else
        omega = ones(N,T,L)/L;
    end
    
    theta = zeros(N,T,L);
    
    %initialize active variances with pre-defined inputs or defaults
    if isfield(optEM,'active_var')
        if (size(optEM.active_var,2) > 1)
            phi = zeros(1,T,L);
            phi(1,:,:) = optEM.active_var';
        else
            phi(1,1,:) =  optEM.active_var;
        end
    else
        scale_factor = zeros(1,1,L);
        scale_factor(1,1,:) = 1/sqrt(L)*(1:L);
        scale_factor = repmat(scale_factor,[N,T,1]);
        phi = sig_var.*scale_factor;
    end;

    optEM.learn_mean = false;
end;

%Resize all initializations to matrix form for scalar multiplications later
lambda = resize(lambda,N,T,1);
omega = resize(omega,N,T,L);
theta = resize(theta,N,T,L);
phi = resize(phi,N,T,L);

if (size(optEM.noise_var,2) == 1)
    optEM.noise_var = repmat(optEM.noise_var,[M T]);
else
    optEM.noise_var = repmat(optEM.noise_var,[M 1]);
end

return
