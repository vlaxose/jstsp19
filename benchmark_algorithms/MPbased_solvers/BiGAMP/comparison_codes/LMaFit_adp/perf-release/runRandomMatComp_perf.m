function runRandomMatComp_perf(scenario)
%%************************************************************************
%% run random matrix completion problems.
%%
%% NNLS, version 0:
%% Copyright (c) 2009 by
%% Kim-Chuan Toh and Sangwoon Yun
%%
%% modified by Zaiwen Wen
%%************************************************************************


doAPGL = 1;
doLMFIT = 1;
doprint = 0;

%%
%% generate random a test problem
%% n = size of the unknown square matrix M, r = rank of M
%%
ndim  = [1e3, 1e3, 1e3, 5e3, 5e3, 5e3, 1e4, 1e4, 1e4, 2e4, 3e4, 5e4, 1e5];
rrank = [10,  50,  100, 10,  50,  100, 10,  50,  100, 10,  10,  10,  10];
pfac  = [6,   4,   3,   6,   5,   4,   6,   5,   4,   6,   6,   6,   6];
%%
problem_type = 'NNLS';
% problem_type = 'SDLS';
%scenario = 'noiseless';
%scenario = 'noisy';
if nargin < 1
    scenario = 'noiseless';
end
%   randstate_idx = [1:2];

nprob = length(ndim);
randstate_idx = [1:3];
randstate_idx = [1];
%%

% opts.print = 2;

fprintf('Each column of the output corresponds to a column in Table 4.2 in order\n \n')
fprintf('%4s %4s %6s %6s %8s %8s %8s %8s %8s', 'n', 'r', 'SR', 'FR', 'mu', 'iter', 'rank', 'CPU','MSE');
fprintf('%8s %6s %8s %8s', 'iter', 'rank', 'CPU','MSE');
fprintf('%8s %6s %8s %8s\n', 'iter', 'rank', 'CPU','MSE');
           
            
for kk = [1:nprob];
% for kk = 1;
    r  = rrank(kk);
    cp = pfac(kk);
    n  = ndim(kk);
    nr = n; nc = n;
    if strcmp(scenario,'noiseless')
        noiseratio = 0;   %% noiseless case
    else
        noiseratio = 0.1; %% noisy case
    end
    for randstate = [randstate_idx];
        %fprintf('\n Create matrix %2.0d with rank = %2.0d,',randstate,r);
        randn('state',double(randstate));
        rand('state',double(randstate));
        if strcmp(problem_type,'SDLS')
            dr = n*r - r*(r-1)/2;
            p  = 2*cp*dr; %% number of sampled entries
        else
            dr = r*(nr+nc-r);
            p  = cp*dr; %% number of sampled entries
        end
        %% construct M (nxn), Omega (nxn)
        %% M = M.U*M.V';
        %% Omega = spconvert([II,JJ,ones(p,1); n,n,0]);
        %% B = M.*Omega;
        M.U = randn(nr,r);
        if strcmp(problem_type,'SDLS')
            M.V = M.U;
        else
            M.V = randn(nc,r);
        end
        normM = sqrt(sum(sum((M.U'*M.U).*(M.V'*M.V))));
        prob = p/(nr*nc);
        II = zeros(p,1); JJ = zeros(p,1); cnt = 0;
        if strcmp(problem_type,'SDLS')
            for j=1:nc
                tmp = rand(j-1,1);
                idx = find(tmp < prob);
                idxkeep = [idx; j];
                len = length(idxkeep);
                II(cnt+[1:len]) = idxkeep;
                JJ(cnt+[1:len]) = j*ones(len,1);
                cnt = cnt + len;
            end
        else
            for j=1:nc
                tmp = rand(nr,1);
                idx = find(tmp < prob);
                len = length(idx);
                II(cnt+[1:len]) = idx;
                JJ(cnt+[1:len]) = j*ones(len,1);
                cnt = cnt + len;
            end
        end
        II = II(1:cnt); JJ = JJ(1:cnt); p = cnt;
        %%
        Jcol = compJcol(JJ);
        bb = Amap_MatComp(M,II,Jcol); 
        B  = spconvert([II,JJ,bb; n,n,0]);
        if strcmp(scenario,'noiseless')
            xi = sparse(p,1);
            sigma = 0;
        else
            randnvec = randn(p,1);
            sigma = noiseratio*norm(bb)/norm(randnvec);
            xi = sigma*randnvec;
            B  = B + spconvert([II,JJ,xi; nr,nc,0]);
        end
        if strcmp(problem_type,'SDLS');
            B = B+triu(B,1)';
        end
        [II,JJ,bb] = find(B);
        Jcol = compJcol(JJ);
        %%------------------------------------------------
        %% evaluate the regularization parameter mu
        %%
        options.tol = 1e-8;
        mumax = svds(sparse(B),1,'L',options);
        mu_scaling = 1e-4;
        mutarget   = mu_scaling*mumax;
        noiseratio = norm(xi)/norm(bb);
        
        if doAPGL
            tstart = clock;
            %Amap  = @(X) ProjOmega(X,II,Jcol);
            %ATmap = @(y) spconvert([II,JJ,y; nr,nc,0]);
            
            Amap  = @(X) Amap_MatComp(X,II,Jcol);
            if (exist('mexspconvert')==3);
                ATmap = @(y) mexspconvert(nr,nc,y,II,Jcol);
            else
                ATmap = @(y) spconvert([II,JJ,y; nr,nc,0]);
            end
            
            par.tol     = 1e-4;
            par.verbose = 0;
            par.plotyes = 0;
            par.continuation_scaling = mu_scaling;
            par.truncation_start = 10; 
            [X,iter,time,sd,hist]= ...
                APGL(nr,nc,problem_type,Amap,ATmap,bb,mutarget,0,par);
            if isstruct(X)
                normX = sqrt(sum(sum((X.U'*X.U).*(X.V'*X.V))));
                trXM = sum(sum((M.U'*X.U).*(M.V'*X.V)));
            else
                normX = norm(X,'fro'); trXM = sum(sum(M.U.*(X*M.V)));
            end
            mse = sqrt(normX^2+normM^2-2*trXM)/normM;
            runhist.mu(randstate)     = mutarget;
            runhist.mumax(randstate)  = mumax;
            runhist.time(randstate)   = etime(clock,tstart);
            runhist.iter(randstate)   = iter;
            runhist.obj(randstate)    = hist.obj(end);
            runhist.mse(randstate)    = mse;
            runhist.svp(randstate)    = hist.svp(end);
            runhist.maxsvp(randstate) = max(hist.svp);
            runhist.maxsig(randstate) = max(sd);
            runhist.minsig(randstate) = min(sd(find(sd>0)));
            
            fprintf('%d & %d & %4.3f & %4.3f & %3.2e & %4d & %d & %8.2f & %3.2e &', ...
                n, r, p/(n*n), dr/p, mutarget, runhist.iter(randstate), ...
                runhist.svp(randstate), runhist.time(randstate), runhist.mse(randstate));
        end
        
        if doLMFIT
            Known.Ik = II;  Known.Jk = JJ;          m = n;
            
            opts = [];
            opts.tol = 1e-4;
            opts.print = doprint;
            
            estk = floor(1.25*r);
            
            % call solver
            tstart = clock;
            [X,Y,Out] = lmafit_mc_adp(m,n,estk, Known, bb,opts);
            tsolve = etime(clock,tstart);
        
            normX = sqrt(sum(sum((X'*X).*(Y*Y'))));
            trXM = sum(sum((M.U'*X).*(M.V'*Y')));
            mse = sqrt(normX^2+normM^2-2*trXM)/normM;
%             fprintf('%d & %d & %d & %d & %4d & %d & %8.2f & %3.2e\n', ...
%                 n, p, r, p/dr, Out.iter,  Out.rank, tsolve, mse);
            fprintf('%4d & %d & %8.2f & %3.2e & ', ...
                    Out.iter,  Out.rank, tsolve, mse);

            Known.Ik = II;  Known.Jk = JJ;          m = n;
            
            opts = [];
            opts.tol = 1e-4;
            opts.print = 0;
            estk = floor(1.5*r);
            
            % call solver
            tstart = clock;
            [X,Y,Out] = lmafit_mc_adp(m,n,estk, Known,bb,opts);
            tsolve = etime(clock,tstart);
        
            normX = sqrt(sum(sum((X'*X).*(Y*Y'))));
            trXM = sum(sum((M.U'*X).*(M.V'*Y')));
            mse = sqrt(normX^2+normM^2-2*trXM)/normM;
%             fprintf('%d & %d & %d & %d & %4d & %d & %8.2f & %3.2e\n', ...
%                 n, p, r, p/dr, Out.iter,  Out.rank, tsolve, mse);
            fprintf('%4d & %d & %8.2f & %3.2e \\\\ \\hline\n', ...
                    Out.iter,  Out.rank, tsolve, mse);
                
                
        end
        
    end
end
