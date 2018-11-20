%function Test_MC_PhasePlot

doprofile = 0;

m  = 500;  n = m; 
rlist  = [2:3:60];   rlen = length(rlist);
SRlist = [0.01:0.05:0.9];  Slen = length(SRlist);                   

nRand = 50;
nRand = 5;

tolsucc = 1e-3;

stat = zeros(rlen,Slen);
for dr = 1:rlen
    r = rlist(dr);  
    for ds = 1:Slen
        SR = SRlist(ds);  p = floor(SR*m*n); FR = (r*(m+n-r)/p);
        fprintf('\nn: %8d, r: %4d, SR: %2.2f, FR: %2.2f \n', n, r, SR, FR);
        if (FR > 1); fprintf('FR > 1\n'); continue; end
        %fprintf('no. \t iter \t final rank \tCPU  \t  MSE\n');
        fprintf('%4s %8s %12s %8s %10s\n','no', 'iter', 'final rank', 'CPU', 'MSE');

        estat = zeros(nRand,1);
        for di = 1:nRand        
            seed = dr*ds*di*100; 
            rand('state',seed);  randn('state',seed);
            % get problem
            Idx = randperm(m*n); Idx = Idx(1:p); Idx = sort(Idx);
            Ml = randn(m,r); Mr = randn(n,r); Ms = Ml*Mr';
            b = Ms(Idx);  normM = norm(Ms,'fro'); 
            
            %-----------------------------------------
            % LMAFIT
            % problem specification
            opts.tol = 1e-4;
            opts.maxit = 1000;
            opts.Zfull = 1;
            opts.print = 0;
            estk = floor(1.25*r);
            
            % call solver
            tstart = clock;
            [X,Y,Out] = lmafit_mc_adp(m,n,estk,Idx,b,opts);
            tsolve = etime(clock,tstart);
            W = X*Y; mse = norm(W-Ms,'fro')/normM;
            %fprintf(' & %4d & %4d & %8.2f & %3.2e \n', ...
            %    Out.iter,  Out.rank, tsolve, mse);
            fprintf('%3d \t %4d \t %4d \t   %8.2f \t %3.2e \n', ...
                    di, Out.iter, Out.rank, tsolve, mse);            
            estat(di, 1) = mse;
        end    
        stat(dr,ds) = sum(estat <= tolsucc)/nRand;
        if ds>=2 && stat(dr,ds) == 1 && stat(dr,ds-1)==1; stat(dr,ds+1:end) = 1; break; end
    end  
  end

save('result/lmafit_phase.mat', 'rlist', 'SRlist', 'stat');



