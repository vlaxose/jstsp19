% mcGAMP: multiclass classification via sparse multinomial logistic
% regression, using GAMP
%
% syntax: [estFin,out,estHist] = mcGAMP(y,A,opt_mc,opt_gamp)



function [estFin,out,estHist] = mcGAMP(y,A,varargin)
% nargin=0; nargout=0; % uncomment this and comment above to run in m-file mode

if nargin==0,
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % begin demonstration mode %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % handle random seed
    if verLessThan('matlab','7.14')
        defaultStream = RandStream.getDefaultStream; %#ok<GETRS>
    else
        defaultStream = RandStream.getGlobalStream;
    end;
    if 1 % new RANDOM trial
        savedState = defaultStream.State;
        save random_state.mat savedState;
    else % repeat last trial
        load random_state.mat %#ok<UNRCH>
    end
    defaultStream.State = savedState;
    
    % load demonstration scenarios
    DEMO = 3;
    switch DEMO
        case 1 
            D = 4;          % number of classes
            N = 30000;      % weight vector dimension
            M = 300;        % number of training observations
            K = 25;         % weight vector sparsity (must be larger than D based on the way the data is generated)
            Pbayes = 0.1;  % Bayes error rate
        case 2
            D = 10;          % number of classes
            N = 10000;       % weight vector dimension
            M = 500;        % number of training observations
            K = 10;        % weight vector sparsity
            Pbayes = 0.10;  % Bayes error rate
        case 3
            D = 4;          % number of classes
            N = 10000;       % weight vector dimension
            M = 1000;        % number of training observations
            K = 10;        % weight vector sparsity
            Pbayes = 0.1;  % Bayes error rate
    end %
    
    % get default options
    opt_mc = MCOpt();
    
    % set simulation options
    opt_mc.plot_hist = 3; % plot gamp history? (0=no, #=figure number)
    opt_mc.plot_wgt = 2; % plot weight vectors? (0=no, #=figure number)
    opt_mc.verbose = true; % print results?
    
    % set feature processing options
    %     opt_mc.autoBalance = false; % dflt=true
    opt_mc.meanRemove = false; % dflt=true
    opt_mc.colNormalizeNorm = false; % dflt=true
    opt_mc.colNormalizeVar = false;

    opt_mc.verbose = 1;
    
    % change default options
    % set classifier parameters
    opt_mc.prior = 'BG'; % in {'BG','ell1'}
    
    % MMSE initialization options
    opt_mc.initTypeMMSE = 1; % in {0 = zeros, 1 = *scaled k-term approx} *scaled to match mnl likelihood
    opt_mc.priorMeanTypeMMSE = 1; % in {0 = zero mean, 1 = xhat0}
    opt_mc.priorVarTypeMMSE = 1; % in {0 = auto, 1 = match xhat0}
    
    % MAP initialization options
    opt_mc.initTypeMAP = 0; % 0 = initialize at zero vector
    opt_mc.priorLambdaMAP = 1; % 0 = default, 1 = set to have certain pvar, 2 = like 0, but with sparsity = 0
    
    % em-tuning options
    opt_mc.tuneVar = true; 
    opt_mc.tuneSpar = true;
    opt_mc.tuneMean = false; 
    opt_mc.tuneDelay = 10; % dflt=0  
   
    % error evaluation options
    opt_mc.knowEmpPerrFxn = false; % empirical probability of error
    opt_mc.knowThePerrFxn = true; % theoretical probability of error
    % ^(not 100% correct in zscored case, but reasonable approximation)
    
    if opt_mc.knowEmpPerrFxn && (M > 1000 || N > 1000)
        warning('it may take awhile to compute the empirical error rate')
    end
    
    % set exceptions to default GAMP options;
    opt_gamp.tol = 1e-4; % dflt=2e-3
    opt_gamp.nit = 200; % dflt=500
    opt_gamp.verbose = 1;
    opt_gamp.stepMax = 1; % dflt=0.01
    opt_gamp.step = .01;
    opt_gamp.stepIncr = .98; % optional, but seems to help
    opt_gamp.uniformVariance = true;
    opt_gamp.adaptStep = false;
    opt_gamp.adaptStepBethe = false;
    opt_gamp.stepWindow = inf; % this could be set to inf to effectively turn off adaptive stepsizing
    opt_gamp.stepTol = 1e-4;

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % generate data %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [y, A, mu, v] = buildDatasets(D,M,N,K,Pbayes);

    x_bayes = mu;
    
    knowBayes = true;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % end demonstration mode %
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    
elseif nargin==1
    
    error('must specify at least two input arguments')
    
else   % nargin>=2
    
    if nargin == 2
        opt_mc = MCOpt();
        opt_gamp = [];
    elseif nargin == 3
        opt_mc = varargin{1};
        opt_gamp = [];
    else
        opt_mc = varargin{1};
        opt_gamp = varargin{2};
    end
    
    [M,N] = size(A);
    D = numel(unique(y));
    
    % obtain info about empirical error functions, if applicable
    if ~isempty(opt_mc.A_test) && ~isempty(opt_mc.y_test)
        M_test = size(opt_mc.A_test,1);
        y_test = opt_mc.y_test;
        opt_mc.knowEmpPerrFxn = true;  
    end
    
    % obtain info about empirical error functions, if applicable
    if ~isempty(opt_mc.x_bayes) && ~isempty(opt_mc.mu) && ~isempty(opt_mc.v)
        opt_mc.knowThePerrFxn = true;  
        x_bayes = opt_mc.x_bayes;
        mu = opt_mc.mu;
        v = opt_mc.v;
        Pbayes = opt_mc.Pbayes;
        knowBayes = true;
    else
        knowBayes = false;
    end

end;

% start timer
tstart = tic;

% check whether or not we need to compute the GAMP history
computeHist = (opt_mc.plot_hist > 0);
if nargout >=3, computeHist = true; end; % override

% default: initialize sparsity based on info-thy arguments:
% we collect M*log2(D) bits of information, and the sparsity pattern
% alone requires D*log2(N-choose-K) or about D*K*log2(N/K) bits
if M<N % find K<=N such that M < K*log2(N/K)
    K0=1; while (K0<M)&&(1/log2(D)*D*K0*log2(N/K0)<M), K0=K0+1; end
else % solution to above is N
    K0 = N;
end
sparsity0 = K0/N;

% shorthand
SPA = opt_mc.SPA;
verbose = opt_mc.verbose;
meanRemove = opt_mc.meanRemove;
colNormalize = opt_mc.colNormalize;
initTypeSP = opt_mc.initTypeSP;
priorMeanTypeSP = opt_mc.priorMeanTypeSP;
priorLambdaMS = opt_mc.priorLambdaMS;
tuneVar = opt_mc.tuneVar;
tuneMean = opt_mc.tuneMean;
tuneSpar = opt_mc.tuneSpar;
tuneDelay = opt_mc.tuneDelay;
plot_hist = opt_mc.plot_hist;
plot_wgt = opt_mc.plot_wgt;
knowEmpPerrFxn = opt_mc.knowEmpPerrFxn;
knowThePerrFxn = opt_mc.knowThePerrFxn;

if verbose
    fprintf('\n\n************************************************************************\n\n')
    if SPA
        fprintf('beginning SPA-SHyGAMP\n')
    else 
        fprintf('beginning MSA-SHyGAMP\n')
    end
    if knowThePerrFxn
        fprintf('Known theoretical error function\n')
        if meanRemove || colNormalize
           fprintf('warning: preprocessing (mean removal/column normalization) will \n')
           fprintf('cause theoretical error function to be inaccurate!!\n') 
        end
    end
    if knowEmpPerrFxn
        fprintf('Known empirical error function\n')
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% process the training data %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check for real-valued features
if any(~isreal(A))
    error('feature matrix must be real valued')
end

% define linear transform operators
AA = @(x) A*x;
AAh = @(z) A.'*z;
S = A.^2;
SS = @(x) S*x;
SSh = @(z) S.'*z;

if knowEmpPerrFxn
    AA_test = @(x) opt_mc.A_test*x;
end
    
% mean-remove the columns of A (while exploiting sparsity of A)
if meanRemove
    %    colOffset = full(mean(A,1)); % contains empirical mean of each column
    %
    %     % use explicit matrix
    %     A = bsxfun(@plus, A, -colOffset);
    
    colOffset = full(mean(A,1)); % contains empirical mean of each column
    colOffset2 = colOffset.^2;
    AA = @(x) A*x - ones(M,1)*(colOffset*x);
    if knowEmpPerrFxn
        AA_test = @(x) opt_mc.A_test*x - ones(M_test,1)*(colOffset*x);
    end
    AAh = @(z) A.'*z - colOffset.'*sum(z,1);
    if issparse(A)
        SS = @(x) S*x - 2*(A*bsxfun(@times,colOffset.',x)) ...
            + ones(M,1)*(colOffset2*x);
        SSh = @(z) S.'*z - 2*bsxfun(@times,colOffset.',A.'*z) ...
            + colOffset2.'*sum(z,1);
    else % A is not sparse
        Smr = bsxfun(@minus,A,colOffset).^2;
        SS = @(x) Smr*x;
        SSh = @(z) Smr.'*z;
    end
end

% column normalization (in a way that exploits sparse A)
if colNormalize
    % compute variance of each column of A
    col_var2 = 1/(M-1) * (sum(S,1) - 2 * sum(A,1).*mean(A) + M*mean(A).^2);
    colScale = 1./sqrt(col_var2); % per-column scaling weights
    colScale(colScale==inf)=0; % don't invert zero-valued quantities
    colScale2 = colScale.^2;
    % scale columns of feature matrix
    AA = @(x) AA(bsxfun(@times,colScale.',x));
    if knowEmpErrFxn
        AA_test = @(x) AA_test(bsxfun(@times,colScale.',x));
    end
    AAh = @(z) bsxfun(@times,colScale.',AAh(z));
    SS = @(x) SS(bsxfun(@times,colScale2.',x));
    SSh = @(z) bsxfun(@times,colScale2.',SSh(z));
end

% compute class conditional mean (via empirical average)
mu_hat = nan(N,D);
for d = 1:D
    % compute using operators
    se = speye(M);
    se = se(y==d,:);
    mu_hat(:,d) = mean(sparse((AAh(se'))),2);
    
    % compute explicitly
    %      mu_hat(:,d) = mean(A(y==d,:))';
end

%  % set Bayes parameters
if knowBayes
    K_bayes = sum(abs(x_bayes(:))>0)/D;
end

% simple one-shot design (near optimal when M>>N)
x_simp = mu_hat; % simple predictor
clear mu_hat;
z = AA(x_simp);
[~,yhat] = max(z,[],2);
Perr_train_simp = sum(y~=yhat)/M;  % training Perr

if knowThePerrFxn 
    Perr_test_simp_the = testErrorRate(x_simp,mu,v);
end
if knowEmpPerrFxn
    z = AA_test(x_simp);
    [~,yhat] = max(z,[],2);
    Perr_test_simp_emp = sum(y_test~=yhat)/M_test;  % training Perr
end

% k0-term approximation (improvment on one-shot)
[~,indx] = sort(abs(x_simp),1,'descend');
if isnan(K0), K0=ceil(sparsity0*N); end;
x_kterm = zeros(N,D);

for d = 1:D
    indx_nz = indx(1:K0,d); % support
    x_kterm(indx_nz,d) = x_simp(indx_nz,d);
end

z = AA(x_kterm);
[~,yhat] = max(z,[],2);
Perr_train_kterm = sum(y~=yhat)/M;  % training Perr

if knowThePerrFxn 
    Perr_test_kterm_the = testErrorRate(x_kterm,mu,v);
end
if knowEmpPerrFxn
    z = AA_test(x_kterm);
    [~,yhat] = max(z,[],2);
    Perr_test_kterm_emp = sum(y_test~=yhat)/M_test;  % training Perr
end

% % display 'rough' classifiers' performance
% if verbose
%     fprintf('\n')
%     % test error (the/emp)
%     if knowThePerrFxn
%         fprintf('simple classifer theo. Pe = %5.3f\n',Perr_test_simp_the)
%         fprintf('kterm classifer theo. Pe = %5.3f\n',Perr_test_kterm_the)
%         fprintf('\n')
%     end
%     if knowEmpPerrFxn
%         fprintf('simple classifer emp. Pe = %5.3f\n',Perr_test_simp_emp)
%         fprintf('kterm classifer emp. Pe = %5.3f\n',Perr_test_kterm_emp)
%         fprintf('\n')
%     end
% end

%%%%%%%%%%%%%%%%%%
% configure GAMP %
%%%%%%%%%%%%%%%%%%

% set the default GAMP options %
if SPA
    optGAMP = GampOpt();
    optGAMP.verbose = false;
    optGAMP.tol = 1e-3; % dflt=1e-4
    optGAMP.uniformVariance = true; % dflt=false
    optGAMP.step = .025;
    optGAMP.stepIncr = .98; 
    optGAMP.adaptStep = false; 
else
    optGAMP = GampOpt();
    optGAMP.verbose = false;
    optGAMP.tol = 1e-3; % dflt=1e-4
    optGAMP.uniformVariance = true; % dflt=false
    optGAMP.step = .05;
    optGAMP.stepIncr = .98; 
    optGAMP.adaptStep = false; 
end

% override default GAMP options with user-supplied value
if exist('opt_gamp','var') && ~isempty(opt_gamp)
    if isstruct(opt_gamp)
        % user specified a structure of exceptions
        fields = fieldnames(opt_gamp);
    elseif ismethod(opt_gamp,'GampOpt')
        % user specified a full GampOpt object: will completely overwrite defaults!
        fields = properties(opt_gamp);
        warning('Completely overwriting bcGAMP''s GampOpt defaults!')
    else
        error('the 4th input must be an object/structure with GampOpt fields')
    end;
    for i=1:size(fields,1),    % change specified fields
        optGAMP = setfield(optGAMP,fields{i},getfield(opt_gamp,fields{i})); %#ok<SFLD,GFLD>
    end;
end;
optGAMP.legacyOut = false; % mandatory!


if SPA 
    % set initial values for Sum-Product mode
    
    % determine priors which match data and multinomial logistic likelihood
    priorVar0 = setBGPriors(AAh, y, N);
    
    % initialize MMSE GAMP's xhat
    switch initTypeSP
        case 0
            % initialize at zero
            optGAMP.xhat0 = zeros(N,D);
        case 1
            % initialize at x_kterm, scaled to match priors
            optGAMP.xhat0 = x_kterm*sqrt(priorVar0/var(x_kterm(x_kterm~=0)));
    end
    
    % set prior mean and variance type (of non-sparse term)
    switch priorMeanTypeSP
        case 0
            priorMean0 = 0;
        case 1
            priorMean0 = optGAMP.xhat0;
    end
    
    % initialize Bernoulli-Gaussian input estimator
    EstIn0 = AwgnEstimIn(priorMean0,priorVar0,false,'mean0Tune',tuneMean,...
        'autoTune',tuneVar,'counter',tuneDelay);
    EstIn = SparseScaEstim(EstIn0,sparsity0,0,...
        'autoTune',tuneSpar,'counter',tuneDelay);
    
else
    % set initial values for Min-Sum mode
    switch priorLambdaMS
        case 0
            % choose prior mean and variance according to data model,
            % then choose lambda so corresponding laplacian prior has
            % matching variance
            priorVar0 = setBGPriors(AAh, y, N);
            
            priorLambda = sqrt(2./priorVar0); % corresponding Laplacian lambda
        case 1
            % to have a certain pvar
            pvar = 1e0;
            try
                A2 = mean(sum(AA(speye(N)).^2,2));
            catch
                sI = speye(M);
                A2 = nan(min(M,100),1);
                for m = 1:min(M,100)
                    A2(m) = sum(AAh(sI(m,:)').^2);
                end
                A2 = mean(A2);
            end
            priorLambda = 1; % this line is required but has no effect
            optGAMP.xvar0 = pvar/A2;
    end
    
    % initialize ell1 regularizer 
    EstIn = SoftThreshEstimIn(priorLambda,true,'autoTune',tuneSpar,'counter',tuneDelay);
  
end

% initialize multinomial logistic output estimator
EstOut = MultiLogitEstimOut(y,D,~SPA); 

%%%%%%%%%%%%
% run GAMP %
%%%%%%%%%%%%


% set GAMP linear transform object
LinTrans = FxnhandleLinTrans(M,N,AA,AAh,SS,SSh);

if computeHist
    out.gstart = tic;
    [estFin,~,estHist] = gampEst(EstIn, EstOut, LinTrans, optGAMP);
    out.gamp_time = toc(out.gstart);
else
    out.gstart = tic;
    estFin = gampEst(EstIn, EstOut, LinTrans, optGAMP);
    out.gamp_time = toc(out.gstart);
end

% save copy of input estimator
out.EstIn = EstIn;

% stop timer
out.total_time = toc(tstart);

x_gamp = estFin.xhat;

% training error rate
z = AA(x_gamp);
[~,yhat] = max(z,[],2);
Perr_train_gamp = sum(y~=yhat)/M;  % Perr from empirical method

if knowThePerrFxn
    Perr_test_gamp_the = testErrorRate(x_gamp,mu,v);
end

if knowEmpPerrFxn
    z = AA_test(x_gamp);
    [~,yhat] = max(z,[],2);
    Perr_test_gamp_emp = sum(yhat~=y_test)/M_test;
end

if verbose
   if SPA
       fprintf('\nSPA-SHyGAMP finished.\n')
   else
       fprintf('\nMSA-SHyGAMP finished.\n')
   end
   fprintf('Runtime = %.2f seconds.\nTraining-error-rate = %.3f\n',out.gamp_time,Perr_train_gamp)
   if knowThePerrFxn
   fprintf('Test-error-rate (theo) = %.3f\n',Perr_test_gamp_the)
   end
   if knowEmpPerrFxn
       fprintf('Test-error-rate (emp) = %.3f\n',Perr_test_gamp_emp)
   end
end

% plot weight vectors
if plot_wgt
    
    figure(plot_wgt); clf;
   
    [~,indx] = sort(abs(x_simp),'descend');
    indx = indx(1:500,:);
    
    Dmax = 2;
    handy = nan(Dmax,1);
    for d = 1:Dmax
        subplot(Dmax,1,d)
        handy(d) = stem(x_simp(indx(:,d),d),'g+');
        hold on;
        handy(d) = stem(x_kterm(indx(:,d),d),'cx'); set(handy(d),'color',[0,0.5,0]);
        stem(x_gamp(indx(:,d),d),'o--');
        if knowBayes
            stem(x_bayes(indx(:,d),d),'r+-');
        end
        hold off
        if (nargin==0) || (N>1e3), set(gca,'Xscale','log'); end
        xlabel('sorted row index')
        title(strcat('weight vector coefficients for {\bf x}',sprintf('_{%d}',d)))
        if d==1
            if knowBayes
                legend('{\bfX}-simple','{\bfX}-kterm','{\bfX}-GAMP','{\bfX}-Bayes')
            else
                legend('{\bfX}-simple','{\bfX}-kterm','{\bfX}-GAMP')
            end
        end
    end
    drawnow;

end

% plot GAMP history
if plot_hist
    figure(plot_hist); clf;
    
    % compute and plot error-rate versus iteration
    ptrain = nan(size(estHist.it));
    for it = estHist.it'
        x = reshape(estHist.xhat(:,it),[],D);
        z = AA(x);
        [~, yhat] = max(z, [], 2);
        ptrain(it) = sum(y~=yhat)/M;
    end

    plot(estHist.it,ptrain,'r.-');
    axe = axis;
    hold on;
    plot(axe(1:2),Perr_train_kterm*[1,1],'r--',...
        axe(1:2),Perr_train_simp*[1,1],'r:');
    hold off;
    legstr = char('Ptrain GAMP','Ptrain kterm');
    legstr = char(legstr,'Ptrain simple');
    
    if knowThePerrFxn      
        ptest_the = nan(size(estHist.it));
        for it = 1:estHist.it(end)
            x = reshape(estHist.xhat(:,it),[],D);
            ptest_the(it) = testErrorRate(x,mu,v);
        end

        hold on;
        plot(estHist.it,ptest_the,'b.-',...
            axe(1:2),Perr_test_kterm_the*[1,1],'b--',...
            axe(1:2),Perr_test_simp_the*[1,1],'c--');
        hold off;
        legstr = char(legstr,'Ptest GAMP');
        legstr = char(legstr,'Ptest kterm');
        legstr = char(legstr,'Ptest simple');
        
    end
    
    if knowEmpPerrFxn     
        ptest_emp = nan(size(estHist.it));      
        for it = 1:estHist.it(end)   
            x = reshape(estHist.xhat(:,it),[],D);
            z = AA_test(x);
            [~, yhat] = max(z,[],2);
            ptest_emp(it) = sum(y_test~=yhat)/M_test;    
        end
        hold on;
        plot(estHist.it,ptest_emp,'m.-',...
            axe(1:2),Perr_test_kterm_emp*[1,1],'m--',...
            axe(1:2),Perr_test_simp_emp*[1,1],'g--');
        hold off;
        legstr = char(legstr,'Ptest GAMP');
        legstr = char(legstr,'Ptest kterm');
        legstr = char(legstr,'Ptest simple');     
    end

    if knowBayes
        hold on;
        plot(axe(1:2),Pbayes*[1,1],'k');
        hold off;
        legstr = char(legstr,'Bayes');
    end
    legend(legstr,'Location','NE');
    title('Pr(err) vs. Iteration');
    ylabel('test-error-rate')
    xlabel('iteration')
    grid on;
    drawnow;
    
end



























