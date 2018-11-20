% compare the SNIPE estimator to a Bernoulli-Gaussian MMSE estimator with 
% various levels of *overmatched* variance 
% (and sparsity is adjusted to maintain estimator shape in the area of greatest shrinkage)
% Mark Borgerding 2015-01-12

spars = .1; %P(nonzero)
xvar0 = 1; % var(X|nonzero)
N=1e3;
rhat = linspace(-1,1,N)';
rvar = .03;

omega = log( (1/spars-1 )*sqrt(xvar0/rvar) );

XH=[];
l={};


for oclog = 0:4

    oc = 2^oclog;
    ocVar = oc*xvar0;
    ocSpars = 1/(1 +( 1/spars-1 )/ sqrt(oc));

    eo = AwbgnEstimOut(zeros(N,1),ocVar,ocSpars);
    xh =  eo.estim(rhat,rvar);l={l{:} sprintf('xvar/xvarTrue=2^%d',oclog)};
    XH=[XH xh];
end

eo = SNIPEstim(omega);
xh=eo.estim(rhat,rvar);
XH=[XH xh];
l={l{:} 'SNIPE'};

figure(1)
plot(rhat,XH)
legend(l{:},'Location','SouthEast')
xlabel('rhat')
ylabel('xhat')
title('estimator')
grid minor

figure(2)
semilogy(rhat,abs(XH -rhat*ones(1,size(XH,2)) ).^2 )
legend(l{:},'Location','SouthEast')
xlabel('rhat')
ylabel('|xhat-rhat|^2')
title('estimator correction energy')
grid minor
