% EMturboGAMP
%
% EMturboGAMP performs inference in a probabilistic model underlying a
% sparse linear inverse problem in which a length-N unknown sparse signal 
% vector x(t) is transformed by a linear operator A(t) into a length-M 
% "transform vector" z(t), i.e.,
%                    z(t) = A(t)*x(t),    t = 1, ..., T   
% which is observed through a separable noisy channel, yielding a
% length-M vector of real- or complex-valued observations y(t).  The
% relationship between the transform vector, z(t), and the observation
% vector, y(t), is described by the separable probability distribution
%      p(y(t)|z(t)) = p(y(1,t)|z(1,t)) * ... * p(y(M,t)|z(M,t)).
% 
% Note that there are several special cases of this signal model.  The
% first is the standard time-varying CS measurement model:
%              	y(t) = A(t)*x(t) + w(t),  t = 1, ..., T
% where y(t) is a length-M vector of observations, A(t) is an M-by-N
% sensing matrix, x(t) is a length-N unknown signal vector, and w(t) is 
% a length-M vector of corrupting noise.  Another special case is the 
% MMV signal model
%                             Y = A*X + W.
% The single measurement vector (SMV) CS recovery problem is a special case 
% as well, (i.e., T = 1).  Finally, classification can also be performed
% within this framework when y(m) denotes a discrete class label, and
% p(y(m)|z(m)) defines the relationship between the m^th class and the
% inner product between the weight vector, x, and the m^th training sample,
% A(m,:).
%
% EMturboGAMP attempts to recover X by iterative application of the
% Generalized Approximate Message Passing (GAMP) algorithm proposed by
% Sundeep Rangan (see http://arxiv.org/abs/1010.5141) and a turbo message
% passing procedure that accounts for probabilistic structure in the signal
% matrix X.
%
% EMturboGAMP can perform inference under a number of possible signal
% models, wherein X can assume a number of different structured prior
% distributions, and the observation channel, p(y(t)|z(t)), can likewise 
% obey a number of different distributions.  In all cases, the signal model
% is fully described by an object of the TurboOpt class (see TurboOpt.m in 
% the ClassDefs folder), which contains a number of properties that 
% together characterize the statistics of the signal and noise matrices.  
% For further details on how to construct a TurboOpt object to match a 
% desired signal model, please consult the TurboOpt documentation.  If no 
% TurboOpt object is provided, EMturboGAMP will assume a default signal 
% model in which elements of X are distributed as iid Bernoulli-Gaussian 
% random variables, and elements of W are distributed as iid Gaussian 
% random variables.
%
% In addition to performing inference on X, EMturboGAMP can attempt to
% learn the statistics of a particular signal model in an iterative fashion
% through an expectation-maximization (EM) algorithm.  This EM learning
% procedure is highly customizable, and can be configured using the
% TurboOpt object.
%
% EMturboGAMP generates (an approximation to) the conditional mean 
% estimate of X, i.e., E[X | Y], which is returned in the variable Xhat.
% It also generates (approximate) marginal conditional variances for each
% element of X, i.e., var{x(n,t) | Y}, in Xvar.  Lastly, the parameters of
% the signal model, if learned using the EM procedure, are returned in a
% TurboOpt structure.
%
% NOTE: This function relies on several other functions in order to operate
% correctly.  For this reason, the folder ClassDefs must be included in
% MATLAB's path.  Additionally, the GAMP MATLAB package (available at
% http://gampmatlab.sourceforge.net) must be properly installed and
% included in MATLAB's path.
%
% SYNTAX
% [Xhat, Xvar, S_POST, TurboOpt] = EMturboGAMP(Y, A, TurboOpt, GAMPopt, ...
%                                              OutputMask)
%
% INPUTS
% Y             M-by-T observed vector
% A             A 1-by-T cell array of M-by-N measurement matrices, or 
%               objects which inherit from GAMP's LinTrans abstract class.
%               If instead of a cell array, a single M-by-N matrix, (or
%               LinTrans object) is passed, then it is assumed that A(t) =
%               A for all t = 1, ..., T.  (**Note: If A(t) = A for all t,
%               then EMturboGAMP can work more quickly if given a single
%               matrix by running matrix-valued GAMP, oftentimes**)
% TurboOpt      An object of the TurboOpt class, which contains the
%               following properties that define the probabilistic signal
%               model and describe the type of structure in the signal
%               prior (see TurboOpt.m in the ClassDefs folder):
%   .Signal             An object that is a concrete subclass of the 
%                       abstract Signal class, used to define the marginal 
%                       priors of the elements of X.  [Default: A default 
%                       BernGauss object]
%   .Observation        An object that is a concrete subclass of the 
%                       abstract Observation class, used to define the 
%                       observation channel model.  [Default: A default 
%                       GaussNoise object]
%                    ***NOTE: If the property SNRdB is assigned a value in
%                       the GenParams object, this value will override any
%                       model parameter properties assigned in the noise
%                       object. ***
%   .SupportStruct      An object of the SupportStruct class, or of an
%                       inheriting subclass (e.g., MarkovChain1).  This
%                       property is used to specify the form of structure 
%                       that is found in the support pattern of the signal 
%                       X.  [Default: A default SupportStruct object 
%                       (support is iid)]
%   .AmplitudeStruct    An object of the AmplitudeStruct class, or of an
%                       inheriting sub-class (e.g., GaussMarkov).  This
%                       property is used to specify the form of structure
%                       that is found in the amplitudes of the non-zero
%                       elements of the signal X. [Default: A default
%                       AmplitudeStruct object (amplitudes are
%                       independently distributed according to the
%                       distribution of active elements specified in the
%                       Signal class)]
%   .RunOptions         An object of the RunOptions class, which contains
%                       runtime parameters governing the execution of
%                       EMturboGAMP.  [Default: Default RunOptions object]
% GAMPopt       [Optional] An object of the GampOpt class, which can be
%               used to customize various runtime parameters of the GAMP
%               algorithm, including the following properties:
% 	.nit                Number of iterations
%  	.step               Step size
%  	.stepMin            Minimum step size
%  	.stepMax            Maximum step size
%  	.adaptStep          Adaptive step size
% 	.stepWindow         Step size check window size
%  	.bbStep             Barzilai-Borwein step size
%  	.verbose            Print results in each iteration
%  	.tol                Convergence tolerance
% 	.stepTol            Minimum allowed step size
%  	.Avar               Variance in A entries (may be scalar)
% OutputMask    [Optional] If only a fraction of the entries of y(t) are
%               observed at each timestep, an M-by-T logical array may be
%               provided in OutputMask, with OutputMask(m,t) = true
%               implying that Y(m,t) was observed, (false otherwise)
%
% OUTPUTS
% Xhat      	Final signal estimate (N-by-T matrix)
% Xvar         	Variance of each signal component (N-by-T matrix)
% S_POST        Estimated posteriors on the underlying support variables
%               (N-by-T-by-D matrix, where D is the number of non-zero
%               support states)
% TurboOpt      Object of the TurboOpt class, which contains the learned
%               statistics of the signal model (if EM learning was enabled)
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 04/12/12
% Change summary: 
%       - Created (12/10/11; JAZ)
%       - Extended support to time-varying linear operators, A(t)
%         (02/07/12; JAZ)
%       - Added S_POST as an output variable (04/12/12; JAZ)
%       - Added OutputMask as an input (04/18/12; JAZ)
% Version 0.2
%

function [Xhat, Xvar, S_POST, TurboOpt] = ...
    EMturboGAMP(Y, A, TurboOpt, GAMPopt, OutputMask)

%% Set default input values, and initialize signal model parameters

if nargin < 2
    error('EMturboGAMP: Insufficient number of arguments')
end

% Check input A for validity
if isa(A, 'cell')
    commonA = false;    % Probably different A(t)'s for each t
    T = length(A);
    assert(T == size(Y, 2), 'Dimension mismatch betweeen Y and A');
    for t = 1:T
        if isnumeric(A{t})
            try
                A{t} = MatrixLinTrans(A{t});
            catch ME
                error('Verify that GAMP has been included on MATLAB''s path (%s)', ...
                    ME.message)
            end
        elseif ~isa(A{t}, 'LinTrans')
            error(['Each cell of input A must contain either an explicit '...
                'matrix, or an object that inherits from GAMP''s '...
                'LinTrans class'])
        end
    end
elseif isnumeric(A)
    commonA = true;     % A(t) = A for all t = 1, ..., T
    % Convert explicit matrix to a MatrixLinTrans object
    try
        A = MatrixLinTrans(A);
    catch ME
        error('Verify that GAMP has been included on MATLAB''s path (%s)', ...
            ME.message)
    end
elseif isa(A, 'LinTrans')
    commonA = true;     % A(t) = A for all t = 1, ..., T
else
    error(['Input A must either be a 1-by-T cell array, explicit matrix, or ' ...
        'an object that inherits from the GAMP LinTrans class'])
end

% Verify size agreements between A and Y
switch commonA
    case true, [M, N] = A.size();
    case false, [M, N] = A{1}.size();
end
T = size(Y, 2);
assert(size(Y, 1) == M, 'Dimension mismatch betweeen Y and A');

% Determine if an OutputMask is being used
if nargin < 5
    UseMask = false;    % No mask required/provided
else
    assert(all(islogical(OutputMask(:))) || ...
        all(OutputMask(:) == 0 | OutputMask(:) == 1), ...
        'OutputMask must be an M-by-T logical array')
    % User provided a valid mask.  Now check size
    assert(size(OutputMask, 1) == M && size(OutputMask, 2) == T, ...
        'Incorrect size for OutputMask');
    % Go ahead and use the mask
    UseMask = true;
    OutputMask = logical(OutputMask);
end

if nargin < 4 || isempty(GAMPopt)
    GAMPopt = GampOpt();            % Use default GAMP parameters
else
    assert(isa(GAMPopt, 'GampOpt'), 'GAMPopt must be a valid GampOpt object');
end

% Use legacy output format of GAMP
GAMPopt.legacyOut = false;

if nargin < 3 || isempty(TurboOpt)
    try
        TurboOpt = TurboOpt();      % Use default Bernoulli-Gaussian model
    catch ME
        error('The folder ClassDefs must be included in MATLAB''s path (%s)', ...
            ME.message)
    end
else
    assert(isa(TurboOpt, 'TurboOpt'), 'TurboOpt must be a valid TurboOpt object');
end

stop = 0;   % Termination flag
i = 0;      % Iteration counter
Xhat_old = realmin*ones(N,T);   % Will hold previous estimate of X

% Get the needed runtime parameters
[smooth_iters, min_iters, verbose, tol, warm_start] = ...
    getOptions(TurboOpt.RunOptions);

% If not using a common A matrix, replicate the GAMPopt object T times,
% (needed in order to warm-start gampEst)
switch commonA
    case true
        % Nothing to do
    case false
        [tmp{1:T}] = deal(GAMPopt);
        GAMPopt = tmp;
        clear('tmp');
end


%% Begin the turbo message passing procedure

% Start by creating initial EstimIn and EstimOut objects for GAMP
[EstimIn, EstimOut] = TurboOpt.InitPriors(Y, A);

% Allocate storage for GAMP outputs
Xhat = NaN(N,T);  Xvar = NaN(N,T);  Rhat = NaN(N,T);  Rvar = NaN(N,T);
Shat = NaN(M,T);  Svar = NaN(M,T);  Zhat = NaN(M,T);  Zvar = NaN(M,T);
Phat = NaN(M,T);  Pvar = NaN(M,T);

while stop == 0

    % Increment time and check if EM iterations are done
    i = i + 1;
    if i >= smooth_iters
        stop = 1;
    end

    % Run GAMP
    switch commonA
        case true,  % Use matrix-valued GAMP (assuming it's supported)
            % Apply OutputMask if required
            if UseMask
                estFin = gampEst(EstimIn, MaskedEstimOut(EstimOut, ...
                    OutputMask), A, GAMPopt);
            else
                estFin = gampEst(EstimIn, EstimOut, A, GAMPopt);
            end
            [Xhat, Xvar, Rhat, Rvar, Shat, Svar, Zhat, Zvar, Phat, ...
                Pvar] = getState(estFin);
            if isscalar(Xvar)
                % Uniform variance used in GAMP
                Xvar = repmat(Xvar, N, T);
                Rvar = repmat(Rvar, N, T);
            end
        case false,
            % Each timestep has a distinct sensing matrix, A(t), thus we
            % will execute GAMP sequentially for each timestep
            for t = 1:T
                % Apply OutputMask if required
                if UseMask
                    estFin = gampEst(EstimIn{t}, ...
                        MaskedEstimOut(EstimOut{t}, OutputMask(:,t)), ...
                        A{t}, GAMPopt{t});
                else
                    estFin = gampEst(EstimIn{t}, EstimOut{t}, A{t}, ...
                        GAMPopt{t});
                end
                [Xhat(:,t), Xvar(:,t), Rhat(:,t), Rvar(:,t), ...
                        Shat(:,t), Svar(:,t), Zhat(:,t), Zvar(:,t), ...
                        Phat(:,t), Pvar(:,t)] = getState(estFin);
                if isscalar(Xvar(:,t))
                    % Uniform variance used in GAMP
                    Xvar(:,t) = repmat(Xvar(:,t), N, 1);
                    Rvar(:,t) = repmat(Rvar(:,t), N, 1);
                end
            end
    end
	
    % Package GAMP outputs into GAMPState object
    State = GAMPState(Xhat, Xvar, Rhat, Rvar, Shat, Svar, Zhat, Zvar, ...
        Phat, Pvar);
    
    % Warm-start GAMP at next iteration, if user wishes
    if warm_start
        switch commonA
            case true
                GAMPopt.xhat0 = Xhat;
                GAMPopt.xvar0 = Xvar;
                GAMPopt.shat0 = Shat;
                GAMPopt.svar0 = Svar;
            case false
                for t = 1:T
                    GAMPopt{t}.xhat0 = Xhat(:,t);
                    GAMPopt{t}.xvar0 = Xvar(:,t);
                    GAMPopt{t}.shat0 = Shat(:,t);
                    GAMPopt{t}.svar0 = Svar(:,t);
                end
        end
    end
    
    % Use final state of GAMP to create new EstimIn and EstimOut objects by
    % accounting for probalistic structure in X and W.  Also refine EM
    % parameter estimates, if user has indicated to do so.
    [EstimIn, EstimOut, S_POST] = ...
        TurboOpt.UpdatePriors(State, Y, EstimIn, EstimOut, A);
    
    % Report 5-number summary for both X and S if running verbosely, and EM
    % learning updates
    if verbose
        fprintf('**************************************************\n')
        fprintf('***        EMturboGAMP 5-Number Summary        ***\n')
        fprintf('**************************************************\n')
        fprintf('Iteration: %d\n', i)
        fprintf(['Xhat   : Min = %1.2g  |  1st Qrtl = %1.2g  |  ' ...
            'Median = %1.2g  |  3rd Qrtl = %1.2g  |  Max = %1.2g\n'], ...
            quantile(Xhat(:), [0, 0.25, 0.5, 0.75, 1]))
        fprintf(['Xvar   : Min = %1.2g  |  1st Qrtl = %1.2g  |  ' ...
            'Median = %1.2g  |  3rd Qrtl = %1.2g  |  Max = %1.2g\n'], ...
            quantile(Xvar(:), [0, 0.25, 0.5, 0.75, 1]))
        fprintf(['S_POST : Min = %1.2g  |  1st Qrtl = %1.2g  |  ' ...
            'Median = %1.2g  |  3rd Qrtl = %1.2g  |  Max = %1.2g\n'], ...
            quantile(S_POST(:), [0, 0.25, 0.5, 0.75, 1]))
    end
    
    % Get the EM learning report, and print to screen if desired
    EMReport = TurboOpt.EMreport(verbose);
    
    % Check for convergence
    if norm(Xhat_old - Xhat, 'fro') / norm(Xhat_old) < tol && i >= min_iters
        if verbose, fprintf('Terminating early...\n'); end
        stop = 1;
    end
    
    % Check for misconvergence
    if any(isnan(Xhat(:)))
        fprintf('EMturboGAMP diverged :o\n')
        return;
    end

    % Move current estimate to old
    Xhat_old = Xhat;
    
    % Clear any early termination flags if minimum number of smoothing
    % iterations haven't been completed
    if stop == 1 && i < min_iters
        stop = 0;
    end
end;

end     % End of main EMturboGAMP function



function [Xhat, Xvar, Rhat, Rvar, Shat, Svar, Zhat, Zvar, Phat, Pvar] = ...
   getState(estFin)

            Xhat = estFin.xhat;
            Xvar = estFin.xvar;
            Rhat = estFin.rhat;
            Rvar = estFin.rvar;
            Shat = estFin.shat;
            Svar = estFin.svar;
            Zhat = estFin.zhat;
            Zvar = estFin.zvar;
            Phat = estFin.phat;
            Pvar = estFin.pvar;
end
