% script for tesing the matrix completion code lmafit_mc on videos

clear all

doAPGL = 1;
doLmafit = 1;

doprint = 0;
% tol = 1e-4;
% est_rank = 0;   estk = 100; k = 100; rank_max = k;
% est_rank = 2;   estk = 5;   k = 50; rank_max = k;
% tol = 1e-4; est_rank = 0;   k = 50;     estk = k; rank_max = k;
% 
% %tol = 1e-3; est_rank = 0;   estk = 100;   k = 100; rank_max = k;
% tol = 5e-4; est_rank = 2;   estk = 50;   k = 120; rank_max = k;
% tol = 5e-4; est_rank = 2;   estk = 20;   k = 100; rank_max = k;
% tol = 5e-4; est_rank = 2;   estk = 100;   k = 150; rank_max = k;
% 
% tol = 1e-3; est_rank = 2;   estk = 20;   k = 100; rank_max = k;

tol = 1e-3; est_rank = 2;   estk = 20;   k = 80; rank_max = k;

%tol = 1e-4; est_rank = 0;   estk = 120;   k = 120; rank_max = k;

SR = 0.5; %sampling ratio

vidlist = [2]; 
%vidlist = [1];
%vidlist = [1:2];
mvid = length(vidlist);

rand('state',2009);     randn('state',2009);
sigma = 0e-2;

for di = 1:mvid
    vid = vidlist(di);
    switch vid
        case 1; load rhinos;
        case 2; load xylo;
        otherwise;
            error('vid must be 1 or 2');
    end
    Mo = Mo';
    [m, n] = size(Mo);
    fprintf('matrix size: %i x %i\n',m,n);
    
    r = k;
    fprintf('k = %i, noise sigma: %6.2e\n',k,sigma);
    
    p = floor(SR*m*n); FR = (r*(m+n-r)/p);
    if (FR > 1); error('FR > 1'); end
    
    % uniformly sample each column
    dp = floor(SR*n);
    TT = false(m,n);
    
    % generate problem data
    fprintf('Generating ...');
    for di = 1:m
        Idx = randperm(n); Idx = Idx(1:dp); TT(di,Idx) = true;
    end
    Known = find(TT); L = length(Known);
    
    % generate data
    data = double(Mo(Known));
    
    % add noise
    if sigma > 0
        noise = randn(size(data));
        level = sigma*norm(data)/norm(noise);
        data = data + level*noise;
        clear noise;
    end
    
    fprintf(' Done. sr = %6.4f\n',L/(m*n));
    
    % error calculation functions
    %lrdot = @(X1,Y1,X2,Y2) sum(sum((X1'*X2).*(Y1*Y2')));
    ferr = @(u,v) norm(u(:)-v(:))/norm(v(:));

    Mn = zeros(size(Mo)); Mn(:) = 0;
    Mn(Known) = data; Mn = Mn';

    %---------------------------------------------
    % APGL
    if doAPGL == 1
        [II,JJ] = find(TT'); bb = Mn(TT');
        Jcol = compJcol(JJ);
        
        options.tol = 1e-8;
        mumax = svds(Mn,1,'L',options);
        mu_scaling = 1e-4;
        mutarget   = mu_scaling*mumax;

        Amap  = @(X) Amap_MatComp(X,II,Jcol);
        if (exist('mexspconvert')==3)
            ATmap = @(y) mexspconvert(n,m,y,II,Jcol);
        else
            ATmap = @(y) spconvert([II,JJ,y;n,m,0]);
        end

        par.tol     = 1e-3;
        par.maxiter = 1000;
        par.verbose = 1;
        par.plotyes = 0;
        par.truncation = 1;
        par.truncation_gap = 20;
        par.continuation_scaling = mu_scaling;
        par.maxrank = 80;
        problem_type = 'NNLS';
        tic;
        [X,iter,time,sd,hist]= ...
            APGL(n,m,problem_type,Amap,ATmap,bb,mutarget,0,par);
        tsolve = toc; 
        if isstruct(X)
            McA = X.V*X.U';
        else
            McA = X; 
        end
        
        save(strcat('APGL-mov',num2str(vid)), 'McA', 'Mn');
        errAll   = ferr(McA,single(Mo));

%         fprintf('& %3.2e & %4d & %4d & %8.2f  & %3.2e ', ...
%             mutarget, iter,  hist.svp(end), tsolve,  errAll);
        
        fprintf('\nAPGL: mu: %3.2e, iter: %4d, rank: %4d, cpu: %8.2f, err: %3.2e \n', ...
            mutarget, iter,  hist.svp(end), tsolve,  errAll);
        
    end

    if doLmafit == 1
        % problem specification
        opts = [];
        opts.tol = tol;
        opts.maxit = 500;
        opts.Zfull = 1;
        opts.DoQR  = 1;
        opts.print = doprint;
        opts.est_rank = est_rank;
        opts.rank_max = rank_max;
    
        % call solver
        tic; [X,Y, Out] = lmafit_mc_adp(m,n,estk,Known,data,opts); tsolve = toc;
        
        hist_err = 0;
        if hist_err > 0
            semilogy(1:Out.iter,Out.obj,'b.-'); pause(1)
            set(gca,'fontsize',16);
            xlabel('Iteration'); ylabel('Objective');
        end
        
        % compute relerr for the data part
        Mc = single(X*Y); %clear X Y;
        comp = Mc(Known);  
        errKnown = ferr(comp, data); 
        errAll   = ferr(Mc,single(Mo));
        %fprintf('\n\nRelErr_Known = %8.3e\n', errKnown);
        %fprintf('RelErr_All   = %8.3e\n', errAll);
        %fprintf('& %4d & %4d & %8.3f & %3.2e \n', ...
        %    Out.iter, min(rank(X), rank(Y)), tsolve,  errAll);

        fprintf('\nLMAFIT: iter: %4d, rank: %4d, cpu: %8.3f, err: %3.2e \n', ...
            Out.iter, min(rank(X), rank(Y)), tsolve,  errAll);

        
        Mc = Mc'; 
        save(strcat('res-mov',num2str(vid), 'est',num2str(est_rank), 'rank', num2str(estk), 'max', num2str(rank_max)), 'Mc', 'Mn');
    end

    
    
    % clear data comp Known
    
%    Mo = Mo'; Mc = Mc'; Mn = Mn';
%    movo = mat2mov(Mo,imsize,0); %clear Mo
%    movn = mat2mov(Mn,imsize,0); %clear Mn
%    movc = mat2mov(Mc,imsize,0); %clear Mc
   
    
%     hf = figure;
%     % resize figure based on frame's w x h, and place at (150, 150)
%     set(hf, 'position', [150 150 imsize(2) imsize(1)]);
%     axis off
%     % tell movie command to place frames at bottom left
%     movie(hf,movc,10,30,[0 0 0 0]);
    
end
