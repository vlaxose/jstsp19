function [estFin,optFin,EMoptFin,estHist]...
    = EMPBiGAMP(Y,problem,opt,EMopt)

%EMPBiGAMP:  Expectation-Maximization Parametric Bilinear Generalized AMP
%
%EM-P-BiG-AMP tunes the parameters of the distributions on B, C, and Y|Z
%assumed by P-BiG-AMP using the EM algorithm.
%The noise is assumed to be AWGN (or CAWGN).
%
% INPUTS:
% -------
% Y: the noisy data matrix. (M x 1)
% problem: An objet of the class PBiGAMPProblem specifying the problem
%   setup.
% opt (optional):  A set of options of the class PBiGAMPOpt.
% EMopt (optional): A structure containing several fields. If the input is
%   omitted or empty, the values in defaultOptions defined below will be
%   used. The user may pass a structure containing any or all of these fields
%   to replace the defaults. Fields not included in the user input will be
%   replaced with the defaults
%
% OUTPUTS:
% --------
% estFin: Structure containing final P-BiG-AMP outputs
% optFin: The PBiGAMPOpt object used
% EMoptFin: The EM options used
% estHist: Structure containing per iteration metrics about the run

%% Handle Options

%Create options if not provided
if (nargin < 3)
    opt = [];
end

%Handle empty options object
if (isempty(opt))
    opt = BiGAMPOpt();
end

%If no inputs provided, set user options to empty
if nargin < 4
    EMopt = [];
end


%Get problem dimensions
M = problem.M;
Nb = problem.Nb;
Nc = problem.Nc;


%Noise parameters
defaultOptions.noise_var = []; %initial noise variance estimate
defaultOptions.learn_noisevar = true; %learn the variance of the AWGN
defaultOptions.SNR0 = 100;

%B parameters
%Type options are:
%G: Gaussian
%CG: Complex Gaussian
%BG: Bernoulli-Gaussian
%BCG: Bernoulli-CGaussian
%Donut: Donut-prior in complex plane
defaultOptions.B_type = 'G';
defaultOptions.B_var = [];
defaultOptions.B_mean = 0;
defaultOptions.B_lambda = []; %Sparsity rate on [0,1]
defaultOptions.B_learn_mean = false;
defaultOptions.B_learn_var = false;
defaultOptions.B_learn_lambda = false;

%C parameters
%Type options are:
%G: Gaussian
%CG: Complex Gaussian
%BG: Bernoulli-Gaussian
%BCG: Bernoulli-CGaussian
%Donut: Donut-prior in complex plane
%LowRankPlusSparse: C is a combination of a Complex Gaussian and a CBG
%components. The additional option .Csizes is included to specify the
%breakdown between the two.
defaultOptions.C_type = 'G';
defaultOptions.C_var = [];
defaultOptions.C_var_init = 'dense'; %Choose 'dense', 'fixed'
defaultOptions.C_mean = 0;
defaultOptions.C_lambda = []; %Sparsity rate on [0,1]
defaultOptions.C_learn_mean = false;
defaultOptions.C_learn_var = true;
defaultOptions.C_learn_lambda = false;


%Iteration control
defaultOptions.maxEMiter = 20; %maximum number of EM cycles
defaultOptions.tmax = 20; %first EM iteration to use full expression for noise variance update

%Tolerances
defaultOptions.EMtol = opt.tol; %convergence tolerance
defaultOptions.maxTol = 1e-4;%largest allowed tolerance for a single EM iteration

%Init method. Options are {mean, random}
defaultOptions.B_init = 'random';
defaultOptions.C_init = 'random';

%Warm starting, set to true to enable. Warm starting may perform
%faster, but somewhat increases the likelihood of getting stuck
defaultOptions.warm_start = false;


%Combine user options with defaults. Fields not provided will be given
%default values
EMopt = checkOptions(defaultOptions,EMopt);



%% Initialize Noise parameters


%Set noise variance if requested
if isempty(EMopt.noise_var)
    EMopt.noise_var = sum(abs(Y(:)).^2)/(M*101);
end


%Set initial noise variance
nuw = EMopt.noise_var;


%% Initialize B

%Set B Mean
if isempty(EMopt.B_mean)
    B_mean = 0;
else
    B_mean = EMopt.B_mean;
end

%Set B var
if isempty(EMopt.B_var)
    B_var = 1;
else
    B_var = EMopt.B_var;
end

%Set lambda (note that this is not used by some distributions)
if isempty(EMopt.B_lambda)
    B_lambda = 0.1*ones(Nb,1);
else
    B_lambda = EMopt.B_lambda;
end

%Init variance
bvar = B_var;

%Initialize mean
switch EMopt.B_init
    
    case 'mean'
        if numel(B_mean) < Nb
            bhat = B_mean*ones(Nb,1);
        else
            bhat = B_mean;
        end
        
    case 'random'
        warning('Sloppy')
        bhat = sqrt(B_var)*randn(Nb,1);
        
    otherwise
        error('Init method not recognized')
        
end


%% Initialize C

%Set C Mean
if isempty(EMopt.C_mean)
    C_mean = 0;
else
    C_mean = EMopt.C_mean;
end

%Set lambda (note that this is not used by some distributions)
if isempty(EMopt.C_lambda)
    C_lambda = 0.1*ones(Nc,1);
else
    C_lambda = EMopt.C_lambda;
end

%Set C var
if isempty(EMopt.C_var)
    switch EMopt.C_var_init
        
        case 'fixed'
            C_var = 1;
            
        case 'dense'
            
            %Estimate Frobenius norm of A
            Aop = problem.zObject.getAOperator(bhat);
            Aop.estFrob;
            
            %Compute initial variance
            C_var = (norm(Y,'fro')^2 - M*L*nuw) / ...
                (Aop.FrobNorm^2 * L * mean(C_lambda(:)));
            
            
        otherwise
            error('Var init method not recognized')
    end
else
    C_var = EMopt.C_var;
end



%Init variance
cvar = C_var;

%Initialize mean
switch EMopt.C_init
    
    case 'mean'
        if numel(C_mean) < Nc
            chat = C_mean*ones(Nc,1);
        else
            chat = C_mean;
        end
        
    case 'random'
        warning('Sloppy')
        chat = sqrt(C_var)*randn(Nc,1);
        
    otherwise
        error('Init method not recognized')
        
end



%% Options


%Set initializations
opt.bhat0 = bhat;
opt.chat0 = chat;
opt.bvar0 = bvar;
opt.cvar0 = cvar;

%Ensure that diagnostics are off to save run time
opt.diagnostics = 0;

%Ensure that EM outputs are calculated
opt.saveEM = 1;

%Original tolerance
tol0 = opt.tol;

%% Loop Control

%Specify tmax
if ~isfield(EMopt,'tmax') || isempty(EMopt.tmax)
    tmax = EMopt.maxEMiter;
else
    tmax = EMopt.tmax;
end

%If warm starting, enable state-saving in estFin
if EMopt.warm_start
    opt.saveState = true;
end
    

%Initialize loop
t = 0;
stop = 0;

%History
histFlag = false;
if nargout >=4
    histFlag = true;
    estHist.errZ = [];
    estHist.errB = [];
    estHist.errC = [];
    estHist.val = [];
    estHist.step = [];
    estHist.pass = [];
    estHist.timing = [];
end;

%Outer loop for rank learning
SNR = EMopt.SNR0;
zhatOld = 0;

%Build options for learning B and C
Bopts.learn_var = EMopt.B_learn_var;
Bopts.learn_mean = EMopt.B_learn_mean;
Bopts.learn_lambda = EMopt.B_learn_lambda;
Bopts.sig_dim = 'joint';
Copts.learn_var = EMopt.C_learn_var;
Copts.learn_mean = EMopt.C_learn_mean;
Copts.learn_lambda = EMopt.C_learn_lambda;
Copts.sig_dim = 'joint';


%% Main Loop

%EM iterations
%The < 2 condition is so that we can do a final iteration using the full
%noise varaince update without repeating code
while stop < 2
    
    %Safety kill
    if SNR < 1
        break;
    end
    
    %Start timing
    tstart = tic;
    
    %Estimate SNR
    SNRold = SNR;
    if t > 0
        SNR = norm(zhat,'fro')^2 / norm(Y - zhat,'fro')^2;
    end
    
    %Set tolerance
    %tolNew = 1 / SNR;
    tolNew = min(max(tol0,1/SNR),EMopt.maxTol);
    opt.tol = tolNew;
    
    %Increment time exit loop
    t = t + 1;
    if t >= EMopt.maxEMiter || stop > 0
        stop = stop + 1;
    end
    
    %Prior on B
    switch EMopt.B_type
        
        case 'G' %Gaussian
            gB = AwgnEstimIn(B_mean,B_var);
            B_lambda = 1;
        case 'CG' %Complex Gaussian
            gB = CAwgnEstimIn(B_mean,B_var);
            B_lambda = 1;
        case 'BG' %Bernoulli-Gaussian
            if any(B_mean(:) ~= 0)
                gBbase = AwgnEstimIn(B_mean, B_var);
                gB = SparseScaEstim(gBbase,B_lambda);
            else
                gB = BGZeroMeanEstimIn(B_var,B_lambda);
            end
        case 'CBG' %Bernoulli-CGaussian
            gBbase = CAwgnEstimIn(B_mean, B_var);
            gB = SparseScaEstim(gBbase,B_lambda);
            
        otherwise,
            error('B type unknown')
            
    end
    
    %Prior on C
    switch EMopt.C_type
        
        case 'G' %Gaussian
            gC = AwgnEstimIn(C_mean,C_var);
            C_lambda = 1;
        case 'CG' %Complex Gaussian
            gC = CAwgnEstimIn(C_mean,C_var);
            C_lambda = 1;
            
        case 'BG' %Bernoulli-Gaussian
            if any(C_mean(:) ~= 0)
                gCbase = AwgnEstimIn(C_mean, C_var);
                gC = SparseScaEstim(gCbase,C_lambda);
            else
                gC = BGZeroMeanEstimIn(C_var,C_lambda);
            end
        case 'CBG' %Bernoulli-CGaussian
            gCbase = CAwgnEstimIn(C_mean, C_var);
            gC = SparseScaEstim(gCbase,C_lambda);
            
        case 'LowRankPlusSparse' %Mix of low rank and sparse
            
            %Ensure mean and variance are correct size
            if length(C_mean) < sum(EMopt.Csizes)
                C_mean = mean(C_mean)*ones(sum(EMopt.Csizes),1);
            end
            if length(C_var) < sum(EMopt.Csizes)
                C_var = mean(C_var)*ones(sum(EMopt.Csizes),1);
            end
            
            %Ensure lambda is correct size
            if length(C_lambda) ~= EMopt.Csizes(2)
                C_lambda = mean(C_lambda)*ones(EMopt.Csizes(2),1);
            end
            
            gC1 = CAwgnEstimIn(C_mean(1:EMopt.Csizes(1)), C_var(1:EMopt.Csizes(1)));
            gC2 = CAwgnEstimIn(C_mean(EMopt.Csizes(1)+1:end), C_var(EMopt.Csizes(1)+1:end));
            gC2 = SparseScaEstim(gC2,C_lambda);
            gC = EstimInConcat({gC1 gC2},EMopt.Csizes);
            
        otherwise,
            error('C type unknown')
            
    end
    
    %Output log likelihood
    if any(~isreal(Y))
        gOut = CAwgnEstimOut(Y,nuw);
    else
        gOut = AwgnEstimOut(Y, nuw);
    end
    
    %Stop timing
    t1 = toc(tstart);
    
    %Run BiG-AMP
    [estFin2,~,estHist2] = ...
        PBiGAMP(gB, gC, gOut, problem, opt);

    % If warm-starting, use current final-state as the next initial-state
    if EMopt.warm_start
        opt.state = estFin2.state;
    end
    
    %Start timing
    tstart = tic;
    
    %Correct cost function
    estHist2.val = estHist2.val -0.5*M*log(2*pi*nuw);
    
    %Report progress
    if histFlag
        error_value = estHist2.errZ(end);
    else
        error_value = nan;
    end
    
    %Show different results in the sparse+lowrank setup
    switch EMopt.C_type
        case{'LowRankPlusSparse'}
            disp(['It ' num2str(t,'%04d')...
                ' C_lambda = ' num2str(mean(C_lambda(:)),'%5.3e')...
                ' C1_var = ' num2str(mean(C_var(1:EMopt.Csizes(1))),'%0.2f')...
                ' C2_var = ' num2str(mean(C_var(EMopt.Csizes(1)+1:end)),'%0.2f')...
                ' tol = ' num2str(opt.tol,'%5.3e')...
                ' SNR = ' num2str(10*log10(SNR),'%03.2f')...
                ' Z_e = ' num2str(error_value,'%05.4f')...
                ' nuw = ' num2str(nuw,'%5.3e')...
                ' numIt = ' num2str(length(estHist2.errZ),'%04d')])
        otherwise
            disp(['It ' num2str(t,'%04d')...
                ' C_lambda = ' num2str(mean(C_lambda(:)),'%5.3e')...
                ' B_lambda = ' num2str(mean(B_lambda(:)),'%5.3e')...
                ' C_var = ' num2str(mean(C_var(:)),'%0.2f')...
                ' tol = ' num2str(opt.tol,'%5.3e')...
                ' SNR = ' num2str(10*log10(SNR),'%03.2f')...
                ' Z_e = ' num2str(error_value,'%05.4f')...
                ' nuw = ' num2str(nuw,'%5.3e')...
                ' numIt = ' num2str(length(estHist2.errZ),'%04d')])
    end
    
    %Compute zhat
    zhat = problem.zObject.computeZ(estFin2.bhat,estFin2.chat);
    
    
    
    %Calculate the change in signal estimates
    norm_change = norm(zhat-zhatOld,'fro')^2/norm(zhat,'fro')^2;
    
    %Check for estimate tolerance threshold
    if (norm_change < max(tolNew/10,EMopt.EMtol)) &&...
            ( (norm_change < EMopt.EMtol) ||...
            (abs(10*log10(SNRold) - 10*log10(SNR)) < 1))
        stop = stop + 1;
    end
    
    %Update noise variance. Include only a portion of the Zvar
    %in beginning stages of EMGMAMP because true update may make it
    %unstable.
    %Learn noise variance
    if EMopt.learn_noisevar
        
        %First, just the term based on the residual
        nuw = norm(Y - zhat,'fro')^2/M;
        
        
        %Then the component based on zvar
        if t >= tmax || stop > 0
            if isscalar(estFin2.zvar)
                nuw = nuw + estFin2.zvar;
            else
                nuw = nuw + sum(estFin2.zvar)/M;
            end
        end
    end
    
    
    %Estimate new B parameters
    switch EMopt.B_type
        case {'CG','CBG'}
            [B_lambda, B_mean, B_var] =...
                CBG_update(estFin2.qhat, estFin2.qvar,...
                B_lambda, B_mean, B_var, Bopts);
        case{'G','BG'}
            [B_lambda, B_mean, B_var] =...
                BG_update(estFin2.qhat, estFin2.qvar,...
                B_lambda, B_mean, B_var, Bopts);
        otherwise,
            error('Not recognized')
    end
    
    
    %Estimate new C parameters
    switch EMopt.C_type
        case {'CG','CBG'}
            [C_lambda, C_mean, C_var] = ...
                CBG_update(estFin2.rhat, estFin2.rvar,...
                C_lambda, C_mean, C_var, Copts);
        case{'G','BG'}
            [C_lambda, C_mean, C_var] =...
                BG_update(estFin2.rhat, estFin2.rvar,...
                C_lambda, C_mean, C_var, Copts);
        case{'LowRankPlusSparse'}
            
            %Handle uniform variance
            if ~opt.uniformVariance
                %First, we learn the parameters for the dense component
                Copts.learn_lambda = false;
                [~, C_mean1, C_var1] = ...
                    CBG_update(estFin2.rhat(1:EMopt.Csizes(1)),...
                    estFin2.rvar(1:EMopt.Csizes(1)),1,...
                    C_mean(1:EMopt.Csizes(1)), C_var(1:EMopt.Csizes(1)), Copts);
                
                %Now the sparse part
                Copts.learn_lambda = EMopt.C_learn_lambda;
                [C_lambda, C_mean2, C_var2] = ...
                    CBG_update(estFin2.rhat(EMopt.Csizes(1)+1:end),...
                    estFin2.rvar(EMopt.Csizes(1)+1:end),C_lambda,...
                    C_mean(EMopt.Csizes(1)+1:end), C_var(EMopt.Csizes(1)+1:end), Copts);
                
                
            else
                %First, we learn the parameters for the dense component
                Copts.learn_lambda = false;
                [~, C_mean1, C_var1] = ...
                    CBG_update(estFin2.rhat(1:EMopt.Csizes(1)),...
                    estFin2.rvar(1)*ones(EMopt.Csizes(1),1),1,...
                    C_mean(1:EMopt.Csizes(1)), C_var(1:EMopt.Csizes(1)), Copts);
                
                %Now the sparse part
                Copts.learn_lambda = EMopt.C_learn_lambda;
                [C_lambda, C_mean2, C_var2] = ...
                    CBG_update(estFin2.rhat(EMopt.Csizes(1)+1:end),...
                    estFin2.rvar(2)*ones(EMopt.Csizes(2),1),C_lambda,...
                    C_mean(EMopt.Csizes(1)+1:end), C_var(EMopt.Csizes(1)+1:end), Copts);
                
                
            end
            %Combine
            C_mean = [C_mean1;C_mean2];
            C_var = [C_var1;C_var2];
            
        otherwise,
            error('Not recognized')
    end
    
    
    %Reinitialize GAMP estimates
    %This was done differently in some version of the EM codes.
    %Could be a place to consider changes if issues are encountered for a
    %given problem
    zhatOld = zhat;
    opt.bhat0 = estFin2.bhat;
    opt.bvar0 = estFin2.bvar;
    opt.chat0 = estFin2.chat;
    opt.cvar0 = estFin2.cvar;
    opt.step = opt.stepMin;%estHist2.step(end);
    
    %Stop timing
    t2 = toc(tstart);
    
    %Output Histories if necessary
    if histFlag
        estHist.errZ = [estHist.errZ; estHist2.errZ];
        estHist.errB = [estHist.errB; estHist2.errB];
        estHist.errC = [estHist.errC; estHist2.errC];
        estHist.val = [estHist.val; estHist2.val];
        estHist.step = [estHist.step; estHist2.step];
        estHist.pass = [estHist.pass; estHist2.pass];
        if t == 1
            estHist.timing = [estHist.timing; t1 + t2 + estHist2.timing];
        else
            estHist.timing = [estHist.timing; t1 + t2 + estHist.timing(end) ...
                + estHist2.timing];
        end
    end
    
end

%% Cleanup

%Update finals
estFin = estFin2;
optFin = opt;
EMoptFin = EMopt;

%Include learned parameters
if histFlag
    estHist.B_mean = B_mean;
    estHist.B_var = B_var;
    estHist.B_lambda = B_lambda;
    estHist.C_mean = C_mean;
    estHist.C_var = C_var;
    estHist.C_lambda = C_lambda;
    estHist.nuw = nuw;
    
end




