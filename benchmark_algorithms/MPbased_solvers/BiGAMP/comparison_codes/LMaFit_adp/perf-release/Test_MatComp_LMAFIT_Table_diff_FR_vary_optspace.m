function Test_MatComp_LMAFIT_Table_diff_FR_vary_optspace

clear all


doAPGL = 0;
doFPCA = 0;
doSVT = 0;
doOptSpace = 1;
doLMAFIT1 = 0;
doLMAFIT2 = 0;
doLMAFIT3 = 0;
doprint = 0;
mlen = 12;
mlist  = 1e3*ones(mlen,1);  nlist = mlist;
rlist  = [10; 10; 10; 10; 50; 50; 50; 50; 100; 100; 100; 100];
SRlist = [0.04; 0.08; 0.15; 0.3; 0.2; 0.25; 0.3; 0.4; 0.35; 0.4; 0.5; 0.55];
SRlist = [0.04; 0.08; 0.15; 0.3; 0.2; 0.25; 0.3; 0.4; 0.35; 0.4; 0.5; 0.55];


% random number
seed = 2009;

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
        Amap  = @(X) ProjOmega(X,II,Jcol);
        ATmap = @(y) spconvert([II,JJ,y; m,n,0]);
        par.tol     = 1e-4;
        par.maxiter = 1000;
        par.verbose = 0;
        par.plotyes = 0;
        par.continuation_scaling = mu_scaling;
        problem_type = 'NNLS';
        tstart = clock;
        [X,iter,time,sd,hist]= ...
            APGL(m,n,problem_type,Amap,ATmap,b',mutarget,0,par);
        tsolve = etime(clock,tstart);
        if isstruct(X)
            X = X.U*X.V';
        end
        mse = norm(X-Ms,'fro')/normM;
        fprintf('%3.2e & %8.2f & %3.2e &', mutarget,  tsolve, mse);
    end
    

    if doFPCA
        maxr = floor(((m+n)-sqrt((m+n)^2-4*p))/2);
        opts = get_opts_FPCA(Ms,maxr,m,n,SR,FR); 
        opts.mu = mutarget;
        tstart = clock;
        Out = FPCA_MatComp(m,n,Idx,b,opts);
        tsolve = etime(clock,tstart);
        W = Out.x; mse = norm(W-Ms,'fro')/normM;
        fprintf('  %8.2f & %3.2e &', tsolve, mse);
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
        fprintf('  %8.2f & %3.2e &', tsolve, mse);
    end

    if doOptSpace
        tol = 1e-4;
        estk = floor(1.25*r);
        niter = 500;
        % Write B to a file 'input'. 
        % The order has to be 'precisely' increasing column indices first 
        % and increasing row indices next.
        % For example 
        % row col value
        %   2   1   0.1
        %   10  1   0.3
        %   21  1   0.2
        %   1   2   0.4
        %   7   2   0.5
        %   8   3   0.6
        % ...
        fid = fopen('input','w');
        for col=1:n
            for row=1:m
                if (B(row,col)~=0)
                    fprintf(fid,'%i\t %i\t %e\n',row,col,full(B(row,col)));
                end
            end
        end
        fclose(fid);
        % Run OptSpace C-version, 'tsolve' is the computation time of
        % OptSpace after reading B from file to completing OptSpace
        command=['./test ' num2str(m) ' ' num2str(n) ' ' num2str(nnz(B)) ' ' num2str(estk) ' input ' num2str(niter) ' ' num2str(tol) ];
        [status tsolve] = unix(command);

        %tstart = clock;
        %[X S Y dist] = OptSpace(B,1.25*,[],tol);
        %[X S Y dist] = OptSpace(B,estk,niter,tol);
        %tsolve = etime(clock,tstart);
        
        % Compute MSE, by reading X,S,Y from 'outputX' 'outputS' 'outputY'
        fid = fopen('outputX', 'r');
        X = fscanf(fid, '%e', [estk m]);
        X = X';
        fclose(fid);
        fid = fopen('outputY', 'r');
        Y = fscanf(fid, '%e', [estk n]);
        Y = Y';
        fclose(fid);
        fid = fopen('outputS', 'r');
        S = fscanf(fid, '%e', [estk estk]);
        S = S';
        fclose(fid);
        W = X*S*Y'; 
        mse = norm(W-Ms,'fro')/normM;
        fprintf(['  ' tsolve(1:end-1) ' & %3.2e & \n'], mse);
    end

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
        fprintf(' %8.2f & %3.2e \\\\ \\hline \n', tsolve, mse);
    end


end

