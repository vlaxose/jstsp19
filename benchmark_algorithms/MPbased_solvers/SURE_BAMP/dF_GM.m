function output = dF_GM(y, tau, sigmaL, sigmaS, p)

% derivative of the MMSE estimator for the 2-state Gaussian mixture


% sigma0=sqrt(sigmaS);
% sigma1=sqrt(sigmaL);
T1=tau+sigmaS;
T2=tau+sigmaL;


%py = @(y) (1-p).*normpdf(y,0,sqrt(tau+sigma0^2)) ...
%           +p.*normpdf(y,0,sqrt(sigma1^2+tau));

py = @(y) (1-p).*normpdf(y,0,sqrt(T1)) ...
           +p.*normpdf(y,0,sqrt(T2));
       
       
%dpy = @(y) (1-p).*(-y./(tau+sigma0^2)).*normpdf(y,0,sqrt(tau+sigma0^2)) ...
%           +p.*(-y./(sigma1^2+tau)).*normpdf(y,0,sqrt(sigma1^2+tau));

dpy = @(y) (1-p).*(-y./T1).*normpdf(y,0,sqrt(T1)) ...
           +p.*(-y./T2).*normpdf(y,0,sqrt(T2));

%ddpy=@(y) -p/(tau+sigma1^2)*normpdf(y,0,sqrt(tau+sigma1^2)).*(1-y/(tau+sigma1^2)) ...
%    -(1-p)/(tau+sigma0^2)*normpdf(y,0,sqrt(tau+sigma0^2)).*(1-y/(tau+sigma1^2));

ddpy=@(y) -p/T2*normpdf(y,0,sqrt(T2)).*(1-y.^2/T2) ...
    -(1-p)/T1*normpdf(y,0,sqrt(T1)).*(1-y.^2/T1);

dF = @(y) 1+ tau*(ddpy(y)./py(y) - (dpy(y)./py(y)).^2);
output=dF(y);