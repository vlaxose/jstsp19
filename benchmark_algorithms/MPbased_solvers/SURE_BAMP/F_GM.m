function x_hat = F_GM(y, tau, sigmaL, sigmaS, p)
% MMSE estimator for 2-state Guassian mixture data
% Chunli  12/11/2013

sigma0=sqrt(sigmaS);
sigma1=sqrt(sigmaL);

P_y = @(y) (1-p).*normpdf(y,0,sqrt(tau+sigma0^2))+p.*normpdf(y,0,sqrt(sigma1^2+tau));
dPdy_y = @(y) (1-p).*(-y./(tau+sigma0^2)).*normpdf(y,0,sqrt(tau+sigma0^2))+p.*(-y./(sigma1^2+tau)).*normpdf(y,0,sqrt(sigma1^2+tau));
MMSE_shrink = @(y) y+tau.*dPdy_y(y)./P_y(y);

x_hat = MMSE_shrink(y);