%function Test_MC_PhasePlot

clear all

m  = 500;  n = m; 
rlist  = [2:3:60];   rlen = length(rlist);
SRlist = [0.01:0.05:0.9];  Slen = length(SRlist);                   

nRand = 50;

tolsucc = 1e-3;

stat = zeros(rlen,Slen);
for dr = 1:rlen
    r = rlist(dr);  
    for ds = 1:Slen
        SR = SRlist(ds);  p = floor(SR*m*n); FR = (r*(m+n-r)/p);
        fprintf('\nn: %8d, r: %4d, SR: %2.2f, FR: %2.2f \n', n, r, SR, FR);
        if (FR > 1); fprintf('FR > 1\n'); continue; end
        
        fprintf('%4s %6s %8s %12s %8s %10s\n','no', 'mu', 'iter', 'final rank', 'CPU', 'MSE');
        
        estat = zeros(nRand,1);
        for di = 1:nRand        
            seed = dr*ds*di*100; 
            rand('state',seed);  randn('state',seed);
            % get problem
            Idx = randperm(m*n); Idx = Idx(1:p); Idx = sort(Idx);
            Ml = randn(m,r); Mr = randn(n,r); Ms = Ml*Mr';
            b = Ms(Idx);  normM = norm(Ms,'fro'); 
            
            [II,JJ] = ind2sub([m n],Idx);
            B  = sparse(m,n); B(Idx) = b;
            [II,JJ,bb] = find(B);
            Jcol = compJcol(JJ);
            
            options.tol = 1e-8;
            mumax = svds(B,1,'L',options);
            mu_scaling = 1e-4;
            mutarget   = mu_scaling*mumax;
            
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
            problem_type = 'NNLS';
            tstart = clock;
            [X,iter,time,sd,hist]= ...
                APGL(m,n,problem_type,Amap,ATmap,b',mutarget,0,par);
            tsolve = etime(clock,tstart);
            if isstruct(X)
                X = X.U*X.V';
            end
            mse = norm(X-Ms,'fro')/normM;
            estat(di, 1) = mse;
            fprintf('%4d & %3.2e & %4d & %4d & %8.2f & %3.2e \n', ...
                di, mutarget, iter,  hist.svp(end), tsolve, mse);

        end    
        stat(dr,ds) = sum(estat <= tolsucc)/nRand;
        if ds>=2 && stat(dr,ds) == 1; stat(dr,ds+1:end) = 1; break; end
    end  
  end

if doLMAFIT  
    save('result/lmafit_phase.mat', 'rlist', 'SRlist', 'stat');
elseif doAPGL
    save('result/APGL_phase.mat', 'rlist', 'SRlist', 'stat');
end


