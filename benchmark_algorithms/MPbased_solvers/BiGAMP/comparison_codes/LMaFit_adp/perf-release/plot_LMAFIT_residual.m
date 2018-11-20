function plot_LMAFIT_residual


doprint = 0;
rlist  = [10; 10; 10; 10; 50; 100;];
SRlist = [0.04; 0.08; 0.3; 0.3; 0.3; 0.3;];
mlen = length(rlist);
mlist  = 1e3*ones(mlen,1);  nlist = mlist;

doLMAFIT1 = 1;
doLMAFIT2 = 1;
seed = 2009;
fprintf('%7s %6s %6s %6s ', 'n', 'r', 'SR', 'FR');
if doLMAFIT1;  fprintf('%12s %8s','CPU','MSE'); end
if doLMAFIT2;  fprintf('%12s %8s','CPU','MSE'); end
fprintf('\n');

nobj1 = [];
nobj2 = [];
% for di = 9
for di = 1:mlen
    m = mlist(di);  n =m;
    r = rlist(di);
    SR = SRlist(di);  p = floor(SR*m*n); FR = (r*(m+n-r)/p);
    if (FR > 1); error('FR > 1'); end
    %fprintf('%8d & %4d & %2.2f & %2.2f \n', n, r, SR, FR);
    fprintf('%8d & %4d & %2.2f & %2.2f ', n, r, SR, FR);
    rand('state',seed);  randn('state',seed);
    Idx = randperm(m*n); Idx = Idx(1:p); Idx = sort(Idx);
    Ml = randn(m,r); Mr = randn(n,r); Ms = Ml*Mr';
    b = Ms(Idx); normM = norm(Ms, 'fro'); 

    %-----------------------------------------
    % LMAFIT
    if doLMAFIT1
        opts.tol = 1e-4;
        opts.maxit = 1000;
        opts.Zfull = 0;
        opts.print = doprint;
        estk = floor(1.25*r);
        tstart = clock;
        [X,Y,Out] = lmafit_mc_adp(m,n,estk,Idx,b,opts);
        tsolve = etime(clock,tstart);
        X = X*Y;  mse = norm(X-Ms,'fro')/normM;
        fprintf('& %8.2f & %3.2e ', tsolve, mse);
        nobj1{di} = Out.obj;
    end
    

    if doLMAFIT2
        opts.tol = 1e-4;
        opts.maxit = 1000;
        opts.Zfull = 0;
        opts.print = doprint;
        estk = floor(1.5*r);
        tstart = clock;
        [X,Y,Out] = lmafit_mc_adp(m,n,estk,Idx,b,opts);
        tsolve = etime(clock,tstart);
        X = X*Y;  mse = norm(X-Ms,'fro')/normM;
        fprintf('& %8.2f & %3.2e ', tsolve, mse);
        nobj1{di} = Out.obj;
    end
    fprintf('\n');


end
save('result/lmafit_residual.mat', 'rlist', 'SRlist', 'nobj1', 'nobj2');

% clear all
% load('result/lmafit_residual.mat', 'rlist', 'SRlist', 'nobj1', 'nobj2');

fig = figure(1);
semilogy(1:length(nobj1{1}), nobj1{1}, '-*', ...
    1:length(nobj1{2}), nobj1{2}, ':d', ...
    1:length(nobj1{3}), nobj1{3}, '-.s','LineWidth',2,'MarkerSize',8)

hl=legend('SR=0.04', 'SR=0.08', 'SR=0.3',1); set(hl,'FontSize',15);
ylabel('normalized residual','fontsize',14); xlabel('iteration','fontsize',14);
set(gca,'FontSize',14)
print(fig , '-depsc','./result/resi-var-SR-fix-r-lmafit1.eps');


fig = figure(2);
semilogy(1:length(nobj1{4}), nobj1{4}, '-*', ...
    1:length(nobj1{5}), nobj1{5}, ':d', ...
    1:length(nobj1{6}), nobj1{6}, '-.s','LineWidth',2,'MarkerSize',8)
hl=legend('r=10', 'r=50', 'r=100',1); set(hl,'FontSize',15);
ylabel('normalized residual','fontsize',14); xlabel('iteration','fontsize',14);
set(gca,'FontSize',14)
print(fig , '-depsc','./result/resi-fix-SR-var-r-lmafit1.eps');


