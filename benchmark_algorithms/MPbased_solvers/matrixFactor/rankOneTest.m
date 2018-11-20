% Add path
addpath('../main');

% Simulation parameters
m = 1000;       % Dimension of u
n = 500;        % Dimemnsion of v
nit = 10;       % number of iterations
newdat = 1;     %
useSE = 0;

% Simulation parameters set in the batch script
if (~exist('runBatch','var'))
    runBatch = 0;
end
if (~runBatch)
    snr = 1;    % SNR in dB
    ntest = 10; % number of test
end

% Get expected correlation
if (useSE)
    load data/rankOneSE;
    [mm, im] = min(abs(snrTest - snr));
    corruSE = corru(:,im);
    corrvSE = corrv(:,im);
end

% Default options
opt = RankOneFitOpt();
opt.nit = nit;

% Distributions
ndist = 2;      % u and v
GAUSS_DIST = 1; % Gaussian
CONST_DIST = 2;
EXP_DIST = 3;
distType = [GAUSS_DIST EXP_DIST];
sparseRat = [1 0.1];

% Estimator methods
LIN_EST = 1;        % rankOneFit with linear estimator
MMSE_EST = 2;       % rankOneFit with MMSE estimator
MAX_SING_EST = 3;   % maximum singular vector
methStr = {'iter-lin', 'iter-mmse', 'max-sing'};
methTest = [LIN_EST MMSE_EST]';
nmeth = length(methTest);


% Generate random vectors for u and v
estim0 = cell(ndist,1);
estim = cell(ndist,1);
for idist = 1:ndist
    
    % Gaussian distribution
    if (distType(idist) == GAUSS_DIST)
        if (idist == 1) % u
            mean0 = 0;
            var0 = 1;
        else
            mean0 = 0.1; % v
            var0 = 1;
        end
        estim0{idist} = AwgnEstimIn(mean0, var0);
    elseif (distType(idist) == CONST_DIST)
        x = [0 1]';
        px = [0.9 0.1]';
        estim0{idist} = DisScaEstim(x,px);
    elseif (distType(idist) == EXP_DIST)
        nx = 100;
        x = linspace(1/nx,2,nx)';
        px = exp(-x);
        px = px/sum(px);
        estim0{idist} = DisScaEstim(x,px);
    end
    if (sparseRat(idist)<1)
        estim{idist} = SparseScaEstim( estim0{idist}, sparseRat(idist) );
    else
        estim{idist} = estim0{idist};
    end
    
end

% Set distributions
estimu = estim{1};
estimv = estim{2};

% Get initial variances
[umean0,uvar0] = estimu.estimInit();
[vmean0,vvar0] = estimv.estimInit();
vsq0 = vmean0^2+vvar0;
usq0 = umean0^2+uvar0;
wvar = usq0*vsq0*10^(-0.1*snr);

% Linear estimator for v
estimvLin = AwgnEstimIn(0, vsq0);

% Initialize vectors
ucorrFinal = zeros(ntest, nmeth);
vcorrFinal = zeros(ntest, nmeth);
ucorrPred = zeros(ntest, nmeth);
vcorrPred = zeros(ntest, nmeth);
ucorrIter = zeros(nit, 2*nmeth, ntest);
vcorrIter = zeros(nit+1, 2*nmeth, ntest);

% Main simulation loop
for itest = 1:ntest
    
    
    % Generate random instance
    if (newdat)
        u0 = estimu.genRand(m);
        v0 = estimv.genRand(n);
        W = sqrt(wvar*m)*randn(m,n);
        A = u0*v0' + W;
    end
    
    
    % Loop over methods
    for imeth = 1:nmeth
        
        meth = methTest(imeth);
        if (meth == MAX_SING_EST)
            
            % Max singular vector
            [ut,st,vt] = svds(A,1);
        else
            
            if (meth == LIN_EST)
                % Iterative with linear
                opt.linEst = 1;              
                [ut,vt,hist] =  rankOneFit(A,estimu,estimv,wvar,opt,u0,v0);
                
            elseif (meth == MMSE_EST)
                % Iterative with MMSE
                opt.linEst = 0;
                opt.minav = 1e-3;                
                [ut,vt,hist] =  rankOneFit(A,estimu,estimv,wvar,opt,u0,v0);
            end
            
            % Compute predicted value
            % Compute E(u0^2) and E(v0^2)
            % Get second-order statistics of u0 and v0
            if (opt.compTrue)
                vsq = [vsq0 norm(v0)^2/n];
                usq = [usq0 norm(u0)^2/m];
            end
            
            % Compute correlation
            ucorrIter(:,[2*imeth-1 2*imeth],itest) = ...
                abs(hist.au1).^2./(hist.au0) ./ repmat(usq, opt.nit, 1);
            vcorrIter(:,[2*imeth-1 2*imeth],itest) = ...
               abs(hist.av1).^2./(hist.av0) ./ repmat(vsq, opt.nit+1, 1);
            ucorrPred(itest,imeth) = ucorrIter(end,2*imeth-1,itest);
            vcorrPred(itest,imeth) = vcorrIter(end,2*imeth-1,itest);
            
            
        end
        
        % Store correlation
        ucorrFinal(itest, imeth) = abs(ut'*u0)^2/norm(u0)^2/norm(ut)^2;
        vcorrFinal(itest, imeth) = abs(vt'*v0)^2/norm(v0)^2/norm(vt)^2;
        
        % Print result
        fprintf(1,'%d %10s u:%f %f v: %f %f\n', itest, methStr{meth}, ...
            ucorrFinal(itest, imeth), ucorrPred(itest,imeth), ...
            vcorrFinal(itest, imeth), vcorrPred(itest,imeth) );
    end 
       
end

% Print result
for imeth = 1:nmeth
    fprintf(1,'Final %10s u:%f %f v: %f %f\n', methStr{meth}, ...
        mean(ucorrFinal(:, imeth)), mean(ucorrPred(:,imeth)), ...
        mean(vcorrFinal(:, imeth)), mean(vcorrPred(:,imeth)) );
end

