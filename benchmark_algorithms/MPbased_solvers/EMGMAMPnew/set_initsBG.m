%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by EMBGAMP.
%
% This function sets initializations for all unspecified BG parameters to
% defaults.
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 6/05/13
% Change summary: 
%   v 1.0 (JV)- First release
%   v 1.1 (JV)- Warning if noise variance set to 0
%   v 1.2 (JV)- Fixed noise_dim and sig_dim problems and cleaned up
%               variable names
%
% Version 1.2

function [lambda, theta, phi, optEM] = set_initsBG(optEM, Y, A, M, N, T)

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
    lambda = min(del*rho_SE,1);
end;

%Initialize active mean parameter
if isfield(optEM,'active_mean')
    theta = optEM.active_mean;
else
    theta = 0;
end

%Initialize active variance parameter
if isfield(optEM,'active_var')
    phi = optEM.active_var;
else
    if strcmp(optEM.noise_dim,'col')
        phi = (sum(abs(Y).^2,1)-M*optEM.noise_var)/sum(A.multSqTr(ones(M,1)))./lambda;
    else
        phi = (norm(Y,'fro')^2/T-M*optEM.noise_var)/sum(A.multSqTr(ones(M,1)))./lambda;
    end
end

%Resize all parameters to N by T matrix
[~,sizeT] = size(lambda);
if sizeT == T
    lambda = repmat(lambda, [N 1]);
else
    lambda= repmat(lambda, [N, T]);
end

[~,sizeT] = size(theta);
if sizeT == T
    theta = repmat(theta, [N 1]);
else
    theta = repmat(theta, [N, T]);
end

[~,sizeT] = size(phi);
if sizeT == T
    phi = repmat(phi, [N 1]);
else
    phi = repmat(phi, [N, T]);
end

if length(optEM.noise_var) == T
    optEM.noise_var = repmat(optEM.noise_var,[M,1]);
end

return
