function Test_MatComp_LMAFIT_Table_diff_FR_vary_perf

clear all

doprofile = 0;

doAPGL = 1;
doFPCA = 1;
doSVT = 0;
doOptSpace = 0;
doLMAFIT1 = 1;
doLMAFIT2 = 1;
doprint = 0;
mlen = 12;
mlist  = 1e3*ones(mlen,1);  nlist = mlist;
rlist  = [10; 10; 10; 10; 50; 50; 50; 50; 100; 100; 100; 100];
SRlist = [0.04; 0.08; 0.15; 0.3; 0.2; 0.25; 0.3; 0.4; 0.35; 0.4; 0.5; 0.55];
SRlist = [0.04; 0.08; 0.15; 0.3; 0.2; 0.25; 0.3; 0.4; 0.35; 0.4; 0.5; 0.55];

seed = 2009;

fprintf('%4s %6s %6s ', 'r', 'SR', 'FR');
if doAPGL;    fprintf('%8s %12s %8s', 'mu', 'CPU','MSE'); end
if doFPCA;    fprintf('%12s %8s','CPU','MSE'); end
if doSVT;     fprintf('%12s %8s','CPU','MSE'); end
if doOptSpace; fprintf('%12s %8s','CPU','MSE'); end
if doLMAFIT1;  fprintf('%14s %8s','CPU','MSE'); end
if doLMAFIT2;  fprintf('%14s %8s','CPU','MSE'); end
fprintf('\n');

for di = 1:mlen
    m = mlist(di);  n =m;
    r = rlist(di);
    SR = SRlist(di);  p = floor(SR*m*n); FR = (r*(m+n-r)/p);
    if (FR > 1); error('FR > 1'); end
    %fprintf('%8d & %4d & %2.2f & %2.2f \n', n, r, SR, FR);
    fprintf('%4d & %2.2f & %2.2f &',  r, SR, FR);
    rand('state',seed);  randn('state',seed);
    Idx = randperm(m*n); Idx = Idx(1:p); Idx = sort(Idx);
    Ml = randn(m,r); Mr = randn(n,r); Ms = Ml*Mr';
    b = Ms(Idx); normM = norm(Ms, 'fro'); 

    %--------------------
    % find a mu for the nuclear norm minimization formulation
    %[II,JJ] = ind2sub([m n],Idx);
    B  = sparse(m,n); B(Idx) = b;
    if doAPGL || doFPCA || doSVT
       [II,JJ,bb] = find(B);
       Jcol = compJcol(JJ);
       
       options.tol = 1e-8;
       mumax = svds(B,1,'L',options);
       mu_scaling = 1e-4;
       mutarget   = mu_scaling*mumax;
    end

    %---------------------------------------------
    % APGL
    if doAPGL
        Amap  = @(X) Amap_MatComp(X,II,Jcol);
         if (exist('mexspconvert')==3); 
            ATmap = @(y) mexspconvert(m,n,y,II,Jcol); 
         else
            ATmap = @(y) spconvert([II,JJ,y; m,n,0]); 
         end
        
        par.tol     = 1e-4;
        par.maxiter = 1000;
        par.verbose = 0;
        par.plotyes = 0;
        par.continuation_scaling = mu_scaling;
        par.truncation_start = 10; 
        problem_type = 'NNLS';
        if doprofile; profile on; end
        tstart = clock;
        [X,iter,time,sd,hist]= ...
            APGL(m,n,problem_type,Amap,ATmap,b',mutarget,0,par);
        tsolve = etime(clock,tstart);

        if isstruct(X)
            X = X.U*X.V';
        end

        mse = norm(X-Ms,'fro')/normM;

        if ~doprofile;
            fprintf(' %3.2e & %8.2f & %3.2e &', mutarget,  tsolve, mse);
		else
            profile off; %profile viewer;
            stat = profile('info');
            for di=1:size(stat.FunctionTable)
                str = stat.FunctionTable(di).FunctionName;
                tt = stat.FunctionTable(di).TotalTime;
                if strcmp(str, 'lansvd')
                    %fprintf('%20s, %10f\n', str, tt);
                    break;
                end
            end
			fprintf('%3.2e & %8.2f & %3.2e & %2.0f%%&', mutarget,  tsolve, mse, (tt/tsolve)*100);
        end

    end
    

    if doFPCA
        maxr = floor(((m+n)-sqrt((m+n)^2-4*p))/2);
        opts = get_opts_FPCA(Ms,maxr,m,n,SR,FR); 
        opts.mu = mutarget;
        if doprofile; profile on; end
        tstart = clock;
        Out = FPCA_MatComp(m,n,Idx,b,opts);
		%Out = FPCA_MatComp_lansvd(m,n,Idx,b,opts);
        tsolve = etime(clock,tstart);

        W = Out.x; mse = norm(W-Ms,'fro')/normM;

        if ~doprofile;
	        fprintf(' %8.2f & %3.2e &', tsolve, mse);
		else

            profile off; %profile viewer;
            stat = profile('info');
            for di=1:size(stat.FunctionTable)
                str = stat.FunctionTable(di).FunctionName;
                tt = stat.FunctionTable(di).TotalTime;
                if strcmp(str, 'LinearTimeSVD')
                    %fprintf('%20s, %10f\n', str, tt);
                    break;
                end
            end
			fprintf(' %8.2f & %3.2e %2.0f%% &', tsolve, mse, (tt/tsolve)*100);
        end

    end

    if doSVT
        pp  = p/(m*n);
        tau = 5*sqrt(m*n); 
        %tau = 1/mutarget;
        delta = 1.2/pp;    
        %delta = 2;    
        maxiter = 500; 
        tol = 1e-4;
        tstart = clock;
        [U,S,V,numiter] = SVT([m n],Idx,b,tau,delta,maxiter,tol);
        tsolve = etime(clock,tstart);
        W = U*S*V'; mse = norm(W-Ms,'fro')/normM;
        fprintf(' %8.2f & %3.2e &', tsolve, mse);
    end

    if doOptSpace
        tol = 1e-8;
        estk = floor(1.25*r);
        niter = 500;
        tstart = clock;
        [X S Y dist] = OptSpace(B,[],[],tol);
        %[X S Y dist] = OptSpace(B,estk,niter,tol);
        tsolve = etime(clock,tstart);
        W = X*S*Y'; mse = norm(W-Ms,'fro')/normM;
        fprintf(' %8.2f & %3.2e & ', tsolve, mse);
    end

    %-----------------------------------------
    % LMAFIT
    if doLMAFIT1
        opts.tol = 1e-4;
        opts.maxit = 1000;
        opts.Zfull = 0;
        opts.print = doprint;
        estk = floor(1.25*r);
        if doprofile; profile on; end
        tstart = clock;
        [X,Y,Out] = lmafit_mc_adp(m,n,estk,Idx,b,opts);
        tsolve = etime(clock,tstart);
        if doprofile; profile off; profile viewer; end
        X = X*Y;  mse = norm(X-Ms,'fro')/normM;
        fprintf(' %8.2f & %3.2e &', tsolve, mse);
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
        fprintf(' %8.2f & %3.2e', tsolve, mse);
    end
    fprintf('\\\\ \\hline\n');


end

