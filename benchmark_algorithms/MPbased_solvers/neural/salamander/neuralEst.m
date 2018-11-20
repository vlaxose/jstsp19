% Main test of neural estimation methods

% Simulation parameters
batchProc = true;   % Enable batch processing from the file neuralSimBatch
ndly = 30;      % number of delay taps in linear filter model
np = 2;         % number of polynomial coeffs in non-linear model
noiseVar = 1;   % noise variance after linear summation
ymax = 5;       % maximum spike count
savedat = true; % save data
spikeStr = 'ch15_c1';   % channel 
dist = 5;       % Runs algorithms on a subset of the pixels in a square
                % of (2*dist+1) x (2*dist + 1) around the central pixel
              

% Parameters set in the batch processing script
if (~batchProc)
    ntrain = 100000; % number of bins used for training 
                     % remaining used for validation.
    noSparse = true; % Ignore sparsity -- reduces estimation to ML  
    loadSTA = false; % Run STA estimate from previous run
    rho = 0.2;       % Sparse probability
end

% Add path for GAMP routines
addpath('..\main');
addpath('..\groupSparse');

% Load data and get dimensions
load data/NeuralStimFull;       % Load the stimulation matrix
cmd = sprintf('load data/sta_%s', spikeStr);        % Load the count
eval(cmd);
ninTot = StimFull.ncol;
nbinsTot = length(cnt);
    
% Methods to test
STA_METH = 1;
GAMP_METH = 2;
methTest = [STA_METH GAMP_METH];
nmeth = length(methTest);
methStr = {'sta', 'gamp'};

% Select subset of pixels over which to perform estimate
Imax = selRedPix(staSub, dist, xpix, ypix);
nin = length(Imax);

% Create reduced matrix
StimRed = StimFull.getSubMatrix(Imax);

% Measure likelihood of null estimators
pspike = mean(cnt);
plike0 = pspike^pspike*(1-pspike)^(1-pspike);

% Set elements to compute estimate over
Itrain = (1:ntrain)';
Ivalid = (ntrain+1:nbinsTot)';

% Generate scalar output function
cnt1 = cnt(Itrain);
nz = 400;
zpt = linspace(-10,5,nz)';
outEstTrain = NeuralOutEst(cnt(Itrain), zpt, noiseVar, ymax);
outEstValid = NeuralOutEst(cnt(Ivalid), zpt, noiseVar, ymax);

% Find optimal constant input
ncnt = length(cnt);
cntHist = mean(repmat(cnt,1,ymax+1) == repmat((0:ymax),ncnt,1))';
plike = exp( outEstTrain.logpyz*cntHist );
[plikeConst,im] = max(plike);
zoptConst = zpt(im);

% Loop over methods
paramMeth = cell(nmeth,1);
plikeMeth = zeros(nmeth,2);
for imeth = 1:nmeth
    
    % Get method
    meth = methTest(imeth);
    fprintf(1,'Testing %s\n', methStr{meth});
    cnt1 = cnt(Itrain);
    
    if (meth == STA_METH)
        % STA
        nlFitOpt.np = np;
        nlFitOpt.niter = 50;
        nlFitOpt.p0 = zoptConst;
        paramSTA = staNL(StimRed,outEstTrain,ndly,nlFitOpt,Itrain);
        if loadSTA
            load paramSTALast;
        else
            save paramSTALast paramSTA;
        end
        param = paramSTA;
        
    elseif (meth == GAMP_METH)
        % GAMP
        gampOpt.nit = 4;
        if (noSparse)
            gampOpt.rho = 1;
        else
            gampOpt.rho = rho;
        end
        [paramGAMP,uhatIter,plikeIt] = neuralGampEst(StimRed,outEstTrain,...
            paramSTA,Itrain,gampOpt);
        param = paramGAMP;
        
    end
    
    % Save parameter.  Note that a copy is made since param is a handle
    paramMeth{imeth} = param.copy();
    
    % Compute the likelihood    
    plikeMeth(imeth,1) = spikeLike(StimRed, outEstValid, param, Ivalid);
    plikeMeth(imeth,2) = spikeLike(StimRed, outEstTrain, param, Itrain);
    
    % Display results 
    fprintf(1,'meth=%s, ntrain=%d valid=%f train=%f\n', methStr{imeth}, ...
        ntrain, plikeMeth(imeth,1), plikeMeth(imeth,2) );
            
end

for imeth = 1:nmeth
    subplot(1,nmeth,imeth);
    paramMeth{imeth}.plotLinWt();
end

if (savedat)
    if (noSparse)
        rhoEff = 1;
    else
        rhoEff = rho;
    end
    cmd = sprintf('save data/neuralSim_%s_rho%d_nt%d plikeMeth ntrain paramMeth', ...
        spikeStr, rhoEff*100, ntrain);
    eval(cmd);
end
    





