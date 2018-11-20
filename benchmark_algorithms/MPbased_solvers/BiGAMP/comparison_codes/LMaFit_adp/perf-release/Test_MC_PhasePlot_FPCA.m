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
        fprintf('%10s %10s\n','CPU', 'MSE');
        estat = zeros(nRand,1);
        for di = 1:nRand        
            seed = dr*ds*di*100; 
            rand('state',seed);  randn('state',seed);
            % get problem
            Idx = randperm(m*n); Idx = Idx(1:p); Idx = sort(Idx);
            Ml = randn(m,r); Mr = randn(n,r); Ms = Ml*Mr';
            b = Ms(Idx);  normM = norm(Ms,'fro'); 
            
            %-----------------------------------------
            maxr = floor(((m+n)-sqrt((m+n)^2-4*p))/2);
            opts = get_opts_FPCA(Ms,maxr,m,n,SR,FR); 

            tstart = clock;
            Out = FPCA_MatComp(m,n,Idx,b,opts);
            tsolve = etime(clock,tstart);
            W = Out.x; mse = norm(W-Ms,'fro')/normM;
            estat(di, 1) = mse;
            fprintf('  %8.2f & %3.2e \n', ...
                 tsolve, mse);
        end    
        stat(dr,ds) = sum(estat <= tolsucc)/nRand;
        if ds>=2 && stat(dr,ds) == 1; stat(dr,ds+1:end) = 1; break; end
    end  
  end

    save('result/lmafit_FPCA.mat', 'rlist', 'SRlist', 'stat');


