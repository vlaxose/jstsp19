% function runRealMatComp_perf
%%************************************************************************
%% run matrix completion problems for real data sets.
%%
%% NNLS, version 0:
%% Copyright (c) 2009 by
%% Kim-Chuan Toh and Sangwoon Yun
%%
%% modified by Zaiwen Wen
%%************************************************************************

clear;
warning off


namelist = {'jester-1','jester-2','jester-3','jester-all','moive-100K','moive-1M','moive-10M'};


doAPGL = 1;
doLMFIT = 1;
doprint = 0;
% est_rank = 1; r = 10;
% est_rank = 1; r = 100;
% est_rank = 2; r = 10; rank_max = 50;
% est_rank = 1; r = 50; rank_max =100;
% %est_rank = 0; r = 50; rank_max =100;
% est_rank = 0; r = 5; rank_max =100;
% est_rank = 1; r = 50; rank_max =50;
% est_rank = 1; r = 20; rank_max =50;

est_rank = 2; r = 1; 
%est_rank = 2; r = 1; 

fprintf('Each column of the output corresponds to a column in Table 4.5 in order\n \n')

fprintf('%10s %10s %12s', 'fname', 'nr', 'nc');
fprintf('%12s %8s %8s %8s %12s %8s', 'mu', 'iter', 'CPU','NMAE', 'relerr', 'rank');
fprintf('%8s %8s %8s %12s %8s\n', 'iter', 'CPU','NMAE', 'relerr', 'rank');


for dataset = [1:7]    

    fname = namelist{dataset};
    load(fname,'M','nr','nc','Msub'); clear Msub;
   % E = (abs(M)>1e-15);

   % [nr, nc] = size(M); nnzE = nnz(E);
   % p = floor(SR*nr*nc);
   % sparsity = nnzE/(nr*nc);
   % if p/nnzE>0.5
   %     p = floor(0.5*nnzE);
   % end

   % %fprintf('%10s, row: %8d, col: %8d, sp(M): %3.2f, nnzE: %8d, p: %8d, p/nnzE: %4.2f\n', ...
   % %        fname, nr, nc, sparsity,nnzE, p, p/nnzE);
   % Idx = find(E);

   % rand('state',seed);  randn('state',seed);
   % pIdx = randperm(nnzE); Idx = sort(Idx(pIdx(1:p))); 
    %continue; 
    % end processing data  
    %-------------------------------------------------

    normB = sqrt(sum(M.*M));
    zerocolidx = find(normB==0);
    if ~isempty(zerocolidx)
        error('***** B has zero columns');
    end
    [m,n] = size(M);
    [II,JJ,Mvec] = find(M);
    Jcol = compJcol(JJ);
    
    %fprintf('%10s & %10d/%10d &, m=%d, n=%d ', fname, nr, nc, m, n);
    fprintf('%10s & %10d/%10d & ', fname, nr, nc);

    if doAPGL
        
        options = [];
        [Umax,Smax,Vmax] = lansvd(sparse(M),1,'L',options);
        mumax      = max(diag(Smax));
        mu_scaling = 1e-3;
        mutarget   = mu_scaling*mumax;
        mutarget   = 1e-4;
        
        Amap  = @(X) Amap_MatComp(X,II,Jcol);
        if (exist('mexspconvert')==3)
            ATmap = @(y) mexspconvert(nr,nc,y,II,Jcol);
        else
            ATmap = @(y) spconvert([II,JJ,y; nr,nc,0]);
        end
        
        par.tol     = 1e-3;
        par.maxiter = 100;
        par.verbose = 0;
        par.plotyes = 0;
        par.truncation = 1;
        par.truncation_gap = 20;
        par.continuation_scaling = mu_scaling;
        if dataset < 5; par.maxrank = 80; else par.maxrank = 100; end
        tstart = clock;
        problem_type = 'NNLS';
        [X,iter,time,sd,hist] = ...
            APGL(nr,nc,problem_type,Amap,ATmap,Mvec,mutarget,0,par);
        runhist.time   = etime(clock,tstart);
        runhist.iter   = iter;
        runhist.obj    = hist.obj(end);
        runhist.mu     = mutarget;
        runhist.mumax  = mumax;
        runhist.svp    = hist.svp(end);
        runhist.maxsvp = max(hist.svp);
        runhist.maxsig = max(sd);
        runhist.minsig = min(sd(find(sd>0)));

		Xvec = Amap_MatComp(X,II,Jcol);
        Rvec = Mvec-Xvec;
        range = max(Mvec)-min(Mvec);
        runhist.MAE    = mean(abs(Rvec));
        runhist.NMAE   = runhist.MAE/range;
        runhist.relerr   = norm(Rvec,'fro')/norm(Mvec,'fro');

        
        fprintf(' %3.2e & %5d & %8.2f & %3.2e & %3.2e & %5d &', ...
             mutarget, runhist.iter, runhist.time, runhist.NMAE, runhist.relerr, runhist.svp);
    end
    
    if doLMFIT

        [Known.Ik, Known.Jk,Mvec] = find(M);
        L = length(Known.Ik);          m = nr; n = nc;
       
        % problem specification
        opts = [];
        opts.tol = 1e-3;
        opts.maxit = 1000;
        opts.Zfull = 0;
        opts.rk_inc = 2;
        opts.est_rank = est_rank;
        if dataset < 5; opts.rank_max = 80; else opts.rank_max = 100; end
        opts.print = doprint;
        estk = r;
        
        % call solver
        tstart = clock;
        [X,Y,Out] = lmafit_mc_adp(m,n,estk,Known,Mvec,opts);
        tsolve = etime(clock,tstart);
        X.U = X;    X.V = Y';
        
		Xvec = Amap_MatComp(X,II,Jcol);
        Rvec = Mvec-Xvec;
        range = max(Mvec)-min(Mvec);
        NMAE    = mean(abs(Rvec))/range;
        relerr  = norm(Rvec,'fro')/norm(Mvec, 'fro');
        
        options = [];
        [uu,sd,vv] = lansvd('matvec','matTvec',m,n,Out.rank,'L', [], X);
        sd = diag(sd); maxsig = max(sd);
        minsig = min(sd(find(sd>0)));
        Out.rank = nnz(sd(sd>1e-8));

        fprintf('%4d & %8.2f & %3.2e & %3.2e & %5d ', ...
            Out.iter, tsolve, NMAE, relerr, Out.rank);
         
    end
    
    fprintf('\\\\ \\hline \n');
    
end
