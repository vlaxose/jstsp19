function Test_LMAFIT_Table_ill_cond(sig_meth)

%sig_meth = 1; % power-law decaying
%sig_meth = 3; % exponential decaying
if nargin < 1
    sig_meth = 1;
end

doAPGL = 1;
doFPCA = 1;
doSVT = 0;
doOptSpace = 0;
doLMAFIT1 = 1;
doLMAFIT2 = 0;
doLMAFIT3 = 1;
doprint = 0;
mlen = 4;
mlist  = 500*ones(mlen,1);  nlist = mlist;
rlist  = [10; 10; 10; 10;];
SRlist = [0.04; 0.08; 0.15; 0.3;];
SRlist = [0.04; 0.08; 0.15; 0.3;];


seed = 2009;

fprintf('Each column of the output corresponds to a column in Table 4.4 in order\n \n')

fprintf('%5s %5s ', 'SR', 'FR');
if doAPGL;    fprintf('%8s %10s %8s %8s', 'mu', 'rank', 'CPU','MSE'); end
if doFPCA;    fprintf('%12s %8s %8s','rank','CPU','MSE'); end
if doLMAFIT1;  fprintf('%12s %8s %8s','rank','CPU','MSE'); end
if doLMAFIT2;  fprintf('%12s %8s %8s','rank','CPU','MSE'); end
if doLMAFIT3;  fprintf('%12s %8s %8s','rank','CPU','MSE'); end
fprintf('\n');

for di = 1:4
    m = mlist(di);  n =m; r = rlist(di);
    SR = SRlist(di);  p = floor(SR*m*n); FR = (r*(m+n-r)/p);
    if (FR > 1); error('FR > 1'); end
    %fprintf('%8d & %4d & %2.2f & %2.2f \n', n, r, SR, FR);
    %fprintf('%4d & %2.2f & %2.2f &',  r, SR, FR);
    fprintf(' %2.2f & %2.2f &',  SR, FR);

    rand('state',seed);  randn('state',seed);
    
    switch sig_meth
        case 1
            cx = 1; beta = 3; xs = cx*[1:n]'.^(-beta);
        case 2
            cx = 1e3; beta = 1.5; xs = cx*[1:n]'.^(-beta);
        case 3
            cx = 1; beta = .3;  xs = cx*exp(-[1:n]'.*beta);
        case 4
            cx = 1; beta = .005; xs = cx*exp(-[1:n]'.*beta);
    end
    
    Idx = randperm(m*n); Idx = Idx(1:p); Idx = sort(Idx);
    Ml = randn(m); Mr = randn(n); Ms = orth(Ml)*diag(xs)*orth(Mr);
    b = Ms(Idx); normM = norm(Ms, 'fro'); 

    %--------------------
    % find a mu for the nuclear norm minimization formulation
    %[II,JJ] = ind2sub([m n],Idx);
    B  = sparse(m,n); B(Idx) = b;
    if doAPGL || doFPCA || doSVT
       [II,JJ,bb] = find(B);
       Jcol = compJcol(JJ);
       
      % options.tol = 1e-8;
      % mumax = svds(B,1,'L',options);
       mumax = 1;
       mu_scaling = 1e-4;
       mutarget   = mu_scaling*mumax;
    end

    %---------------------------------------------
    % APGL
    if doAPGL
        %Amap  = @(X) ProjOmega(X,II,Jcol);
        %ATmap = @(y) spconvert([II,JJ,y; m,n,0]);
        
        Amap  = @(X) Amap_MatComp(X,II,Jcol);
        if (exist('mexspconvert')==3)
            ATmap = @(y) mexspconvert(m,n,y,II,Jcol);
        else
            ATmap = @(y) spconvert([II,JJ,y; m,n,0]);
        end        
        par.tol     = 1e-4;
        par.maxiter = 1000;
        par.verbose = 0;
        par.plotyes = 0;
        par.continuation_scaling = mu_scaling;

        par.truncation = 1;
        par.truncation_gap = 1000;

        problem_type = 'NNLS';
        tstart = clock;
        [X,iter,time,sd,hist]= ...
            APGL(m,n,problem_type,Amap,ATmap,b',mutarget,0,par);
        tsolve = etime(clock,tstart);
        if isstruct(X)
            X = X.U*X.V';
        end
        mse = norm(X-Ms,'fro')/normM;
        fprintf(' %3.2e & %4d & %8.2f & %3.2e & ', mutarget,  hist.svp(end),  tsolve, mse);
    end
    

    if doFPCA
        maxr = floor(((m+n)-sqrt((m+n)^2-4*p))/2);
        opts = get_opts_FPCA(Ms,maxr,m,n,SR,FR); 
        opts.mu = mutarget;
        tstart = clock;
        Out = FPCA_MatComp(m,n,Idx,b,opts);
        tsolve = etime(clock,tstart);
        W = Out.x; mse = norm(W-Ms,'fro')/normM;
        [U1 S1 V1] = svd(W); nrank = nnz(diag(S1)>1e-8);
        fprintf(' %4d &  %8.2f & %3.2e &', nrank, tsolve, mse);
    end

    %-----------------------------------------
    % LMAFIT
    if doLMAFIT1
        opts.tol = 1e-4;
        opts.maxit = 1000;
        opts.Zfull = 0;
        opts.est_rank = 1;
        opts.rank_min= 5;
        opts.print = doprint;
        
        estk = max(floor(0.1*n),10);
        tstart = clock;
        [X,Y,Out] = lmafit_mc_adp(m,n,estk,Idx,b,opts);
        tsolve = etime(clock,tstart);
        X = X*Y;  mse = norm(X-Ms,'fro')/normM;
        fprintf(' %4d & %8.2f & %3.2e & ', Out.rank, tsolve, mse);
    end
    

    if doLMAFIT2
        opts.tol = 1e-4;
        opts.maxit = 1000;
        opts.Zfull = 0;
        opts.rk_jump = 100;
        opts.print = doprint;
        %estk = floor(1.5*r);
        estk = max(floor(0.1*n),10);
        tstart = clock;
        [X,Y,Out] = lmafit_mc_adp(m,n,estk,Idx,b,opts);
        tsolve = etime(clock,tstart);
        X = X*Y;  mse = norm(X-Ms,'fro')/normM;
        %fprintf(' %8.2f & %3.2e \\\\ \\hline \n', tsolve, mse);
        fprintf(' %4d & %8.2f & %3.2e &', Out.rank, tsolve, mse);
    end
    
    %-----------------------------------------
    % LMAFIT
    if doLMAFIT3
        opts.tol = 1e-4;
        opts.maxit = 1000;
        opts.Zfull = 0;
        opts.est_rank = 2;
        opts.rank_max = max(floor(0.1*n),10);
        opts.rk_inc = 1;
        opts.print = doprint;
        estk = 1;
        
        tstart = clock;
        [X,Y,Out] = lmafit_mc_adp(m,n,estk,Idx,b,opts);
        tsolve = etime(clock,tstart);
        X = X*Y;  mse = norm(X-Ms,'fro')/normM;
        %fprintf('%4d & %8.2f & %3.2e & \n', Out.rank, tsolve, mse);
        fprintf(' %4d & %8.2f & %3.2e', Out.rank, tsolve, mse);
    end    
    
    fprintf(' \\\\ \\hline\n');
    

end

