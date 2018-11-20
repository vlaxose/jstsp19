
%a=randn; b=a+rand;
a=0;b=1;

n = 1e3; % # unknowns per trial
m = 100; % # trials
rvar = 1e-2;

X = rand(n,m)*(b-a) + a;
R = X + randn(size(X)) * sqrt(rvar);
XHAT= nan(2,n,m);
XVAR= nan(2,n,m);

ue = { UnifEstimIn(a,b,false) ,  UnifEstimIn(a,b,true) }; % MMSE and MAP

mse = nan(m,2);
for k=1:m
    for uei=1:2
        [xhat,xvar] = ue{uei}.estim(R(:,k),rvar*ones(n,1));
        mse(k,uei) = mean( abs(xhat-X(:,k) ).^2);
        XHAT(uei,:,k) = xhat;
        XVAR(uei,:,k) = xvar;
    end
end
XHAT_MMSE = squeeze(XHAT(1,:,:));
XHAT_MAP = squeeze(XHAT(2,:,:));
XVAR_MMSE = squeeze(XVAR(1,:,:));
XVAR_MAP = squeeze(XVAR(2,:,:));

if mean( mse(:,1) ) > mean(mse(:,2))
    error 'MMSE estimator is not MMSE'
else
    fprintf('MMSE estimator does have lower MSE than MAP esimator\n')
end

figure(1);
subplot(311)
hist(R(:),101);
title('R = X+Noise')
subplot(312)
hist(XHAT_MMSE(:),101);
title('MMSE estimator E(X|R)')
subplot(313)
hist(XHAT_MAP(:),101);
title('MAP estimator argmax_X p(X|R)')

figure(2)
subplot(211)
plot( R(:), [ XHAT_MMSE(:) XHAT_MAP(:)],'.')
xlabel('Estimator Input')
ylabel('Estimator Output')
legend('MMSE','MAP')
subplot(212)
plot( R(:), [ XVAR_MMSE(:) XVAR_MAP(:)],'.')
xlabel('Estimator Input')
ylabel('Estimator Output Variance')
legend('MMSE','MAP')
