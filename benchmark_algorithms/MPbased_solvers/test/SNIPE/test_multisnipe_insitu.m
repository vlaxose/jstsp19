% compare MultiSNIPEstim against a matched prior
% "in-place" in a GAMP system 
n_x = 1e3;% 
L = 2; % number of discrete points (delta pulse) in p_X
delta = .7; % measurement ratio
lam0 = .05; % fraction of X elements that are not in the discrete distribution
n_z = delta * n_x;
SNR = 30;

theta = linspace(-1,1,L)/2; % delta dirac positions
xvar0 = 1;
x = zeros(n_x,1);
nnzx = round(lam0*n_x);
x(randperm(n_x,nnzx)) = randn(nnzx,1)*sqrt(xvar0);
x(x==0) = -.5 + floor(rand(n_x-nnzx,1)*L)/(L-1);

M = randn(n_z,n_x) / sqrt(n_x);
z = M*x;
varw = norm(z)^2/n_z * 10^(SNR/-10);
y = z + randn(size(z))*sqrt(varw);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compare against a matched prior pdf
% ( mixture of multiple delta functions and a Gaussian )
estIn0 = AwgnEstimIn(zeros(n_x,1),xvar0*ones(n_x,1));
estIn1 = DisScaEstim(theta(:),.5*ones(size(theta')));
estInM = MixScaEstimIn(estIn1, 1-lam0, estIn0);

estOut = AwgnEstimOut(y,varw); % The output conditional pdf is the same for Matched or MultiSNIPE

gopt = GampOpt;
gopt.legacyOut=false;
gopt.adaptStep=false;
gopt.uniformVariance=true;
gopt.step=.5;

estFin = gampEst(estInM,estOut,M,gopt);
nmse_matched = norm(estFin.xhat-x)^2/norm(x)^2;
fprintf('Matched prior: nmse=%.2fdB\n', 10*log10( nmse_matched) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%ov = linspace(log(1/delta-1),log(1/lam0-1)+6,25)';
ov = log(5/lam0-1); % could do better, but this will be in the ballpark
nmse = nan(size(ov));
nit = nan(size(ov));
for k=1:length(ov)
    omega = ov(k);
    estInS = MultiSNIPEstim(theta,omega);

    estFin = gampEst(estInS,estOut,M,gopt);
    nmse(k) = norm(estFin.xhat-x)^2/norm(x)^2;
    nit(k) = estFin.nit;
end
fprintf('MultiSNIPEstim : nmse=%.2fdB\n', 10*log10( min(nmse)) );
%semilogy(ov,nmse)
