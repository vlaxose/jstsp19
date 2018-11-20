% function plot_LMAFIT_rank_estimate

doprint = 0;
rlist  = [10; 10;  10];
SRlist = [0.04; 0.08; 0.3;];
mlen = length(rlist);
mlist  = 1e3*ones(mlen,1);  nlist = mlist;

klist = 10:30;  lenk = length(klist);
seed = 2010;

stat = [];

for di = 1:mlen
    m = mlist(di);  n =m;
    r = rlist(di);
    SR = SRlist(di);  p = floor(SR*m*n); FR = (r*(m+n-r)/p);
    if (FR > 1); error('FR > 1'); end
    %fprintf('%8d & %4d & %2.2f & %2.2f \n', n, r, SR, FR);
    %fprintf('%8d & %4d & %2.2f & %2.2f ', n, r, SR, FR);
    rand('state',seed);  randn('state',seed);
    fprintf('n: %8d, r: %4d, SR: %2.2f, FR: %2.2f \n', n, r, SR, FR);
    fprintf('%4s %8s %10s %12s %18s\n','no', 'iter', 'CPU', 'MSE', 'final rank');
    tstat = zeros(lenk,3);
    for dk = 1:lenk
        opts.tol = 1e-4;
        opts.maxit = 1000;
        opts.Zfull = 0;
        opts.print = doprint;
        estk = klist(dk);
        t1 = 0;  niter = 0; nrun = 50; nmse = 0;
        for dj = 1:nrun
            Idx = randperm(m*n); Idx = Idx(1:p); Idx = sort(Idx);
            Ml = randn(m,r); Mr = randn(n,r); Ms = Ml*Mr';
            b = Ms(Idx); normM = norm(Ms, 'fro'); 
            tstart = clock;
            [X,Y,Out] = lmafit_mc_adp(m,n,estk,Idx,b,opts);
            tsolve = etime(clock,tstart);
            t1 = t1 + tsolve;
            niter = niter + Out.iter;
            X = X*Y;  mse = norm(X-Ms,'fro')/normM;
            nmse = nmse + mse;
            %fprintf(' %3d & %8.2f & %3.2e & %d\n', Out.iter, tsolve, mse, Out.rank);
            fprintf('%3d \t %3d \t %8.2f \t %3.2e \t %d\n', dj, Out.iter, tsolve, mse, Out.rank);

        end
        tsolve = t1/nrun;
        niter = niter/nrun;
        nmse = nmse/nrun;
        %fprintf(' %4u & %8.2f & %3.2e \n', round(niter), tsolve, nmse);
        fprintf('\nn: %8d, r: %4d, SR: %2.2f, FR: %2.2f, avg. iter: %d, avg. cpu: %3.2e, avg. MSE: %3.2e\n\n', ...
                    n, r, SR, FR, round(niter), tsolve, nmse);
        
        tstat(dk,1:3) = [niter, tsolve, nmse];
    end
    stat{di} = tstat;

end
save('result/lmafit_rank_est.mat', 'klist', 'rlist', 'SRlist',  'stat');

%clear all
% load('result/lmafit_rank_est.mat',  'klist', 'rlist', 'SRlist',  'stat');
fig = figure(1);
% klist = 10:2:30;
plot(klist, stat{1}(:,1), '-*', ...
   klist, stat{2}(:,1), ':d', ...
   klist, stat{3}(:,1), '-.s','LineWidth',2,'MarkerSize',8)

hl=legend('SR=0.04', 'SR=0.08', 'SR=0.3',0); set(hl,'FontSize',15);
ylabel('iteration number','fontsize',14); xlabel('rank estimation','fontsize',14);
set(gca,'FontSize',14)
print(fig , '-depsc','./result/estk-iter-lmafit1.eps');

fig = figure(2);
plot(klist, stat{1}(:,2), '-*', ...
   klist, stat{2}(:,2), ':d', ...
   klist, stat{3}(:,2), '-.s','LineWidth',2,'MarkerSize',8)

hl=legend('SR=0.04', 'SR=0.08', 'SR=0.3',0); set(hl,'FontSize',15);
ylabel('cpu','fontsize',14); xlabel('rank estimation','fontsize',14);
set(gca,'FontSize',14)
print(fig , '-depsc','./result/estk-cpu-lmafit1.eps');

return;