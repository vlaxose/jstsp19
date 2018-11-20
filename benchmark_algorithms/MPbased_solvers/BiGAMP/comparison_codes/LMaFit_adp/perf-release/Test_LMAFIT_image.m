function Test_LMAFIT_image

doAPGL = 1;
doLMAFIT1 = 1;
doprint = 0;

Mo = im2double(imread('boat.png'));
[m,n] = size(Mo);

rlist = [floor(0.1*n), 40, floor(0.1*n), 40];
rlist = [50, 50, 50, 40];

seed = 2009;


% fprintf('%4s ', 'r');
if doAPGL;    fprintf('%8s %8s %8s %8s %9s', 'mu', 'iter', 'rank', 'CPU','rel.err'); end
if doLMAFIT1;  fprintf('%10s %8s %8s %9s', 'iter', 'rank', 'CPU','rel.err'); end
fprintf('\n');


% for di = 3
for di = 1:3
    r = rlist(di);
    if di == 1 %|| di == 3
        Ms = Mo;
    elseif di >= 2%2 || di == 4
        load('boat-r-40.mat');
    end
    
    SR = 0.5;  p = floor(SR*m*n); FR = (r*(m+n-r)/p);
    if (FR > 1); error('FR > 1'); end
    %fprintf('%8d & %4d & %2.2f & %2.2f \n', n, r, SR, FR);
%     fprintf('%8d & %4d & %2.2f & %2.2f ', n, r, SR, FR);
    
    rand('state',seed);  randn('state',seed);

    % get problem
    
    if di <= 2
        Idx = randperm(m*n); Idx = Idx(1:p); Idx = sort(Idx);
        Mb = zeros(m,n); Mb(Idx) = Ms(Idx);  Idx = find(Mb);
        b = Ms(Idx);
    elseif di >= 3
        Mb = Ms;
        for dj=50:450
            Mb(dj,[dj:dj+20, n-dj:-1:n-dj-20]) = 0;
        end
        Idx = find(Mb);
        b  = Mb(Idx);
    end
   
    %fig = figure(1); imshow(Mb);
    %print(fig , '-depsc',strcat('./result/img-samp',num2str(di),'.eps'));

    %--------------------------------------------- % APGL
    if doAPGL
        [II,JJ] = ind2sub([m n],Idx);
        B  = sparse(m,n); B(Idx) = b;
        [II,JJ,bb] = find(B);
        Jcol = compJcol(JJ);
        
        options.tol = 1e-8;
        mumax = svds(B,1,'L',options);
        mu_scaling = 1e-4;
        mutarget   = mu_scaling*mumax;
        %mutarget = 1e-1;
        
        Amap  = @(X) Amap_MatComp(X,II,Jcol);
        if (exist('mexspconvert')==3)
            ATmap = @(y) mexspconvert(m,n,y,II,Jcol);
        else
            ATmap = @(y) spconvert([II,JJ,y; m,n,0]);
        end
        
        par.tol     = 1e-3;
        par.maxiter = 1000;
        par.verbose = 0;
        par.plotyes = 0;
        par.truncation = 1;
        par.truncation_gap = 20;
        par.continuation_scaling = mu_scaling;
        par.maxrank = 50;
        problem_type = 'NNLS';
        tstart = clock;
        [X,iter,time,sd,hist]= ...
            APGL(m,n,problem_type,Amap,ATmap,b,mutarget,0,par);
        tsolve = etime(clock,tstart);
        if isstruct(X)
            normX = sqrt(sum(sum((X.U'*X.U).*(X.V'*X.V))));
            trXM = sum(sum((Ml'*X.U).*(Mr'*X.V)));
            MM = X.V*X.U';
        else
            MM = X; 
        end
        
        fprintf('& %3.2e & %4d & %4d & %8.2f & %3.2e ', ...
            mutarget, iter,  hist.svp(end), tsolve, norm(MM-Mo,'fro')/norm(Mo,'fro'));
        
        %fig = figure(2); imshow(MM); 
        %print(fig , '-depsc',strcat('./result/img-apgl-',num2str(di),'.eps'));

    end
    
    %-----------------------------------------
    % LMAFIT
    if doLMAFIT1
        % problem specification
        opts.tol = 1e-3;
        opts.maxit = 1000;
        %opts.Zfull = 1;
        opts.est_rank = 2;
        opts.rank_max = r;
        %opts.rk_inc = 5;

        opts.print = doprint;
        estk = floor(1.25*r);
        estk = 20;
        % call solver
        tstart = clock;
        [X,Y,Out] = lmafit_mc_adp(m,n,estk,Idx,b,opts);
        tsolve = etime(clock,tstart);

        MM = X*Y;
        
        fprintf(' & %4d & %4d &  %8.2f & %3.2e', ...
            Out.iter, Out.rank, tsolve, norm(MM-Mo,'fro')/norm(Mo,'fro'));
        
        %fig = figure(3); imshow(MM); 
        %print(fig , '-depsc',strcat('./result/img-lmafit-',num2str(di),'.eps'));
    end
    
    fprintf(' \\\\ \\hline\n');
end

