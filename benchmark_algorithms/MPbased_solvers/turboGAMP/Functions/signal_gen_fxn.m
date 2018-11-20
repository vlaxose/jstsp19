% SIGNAL_GEN_FXN
%
% This function will generate a realization of a desired signal model.  
% It is assumed that the measurement model is a linear process in which 
% M-by-T dimensional measurements, Y, are obtained from an N-by-T signal 
% matrix, X, whose elements are generated from a distribution defined by
% the Signal, SupportStruct, and AmplitudeStruct classes, through the
% Markov process
%                               X -> Z = A*X -> Y,
% where A is an M-by-N measurement matrix, and the relationship between Z
% and Y is governed by the Observation class object, which defines the
% output channel statistics, p(Y(m,t)|Z(m,t))
%
% Problem dimensions, type of random measurement matrix, and SNR are all
% specified by a GenParams object (see GenParams in the ClassDefs folder)
%
% SYNTAX
% [Y, A, X, S, Observation] = signal_gen_fxn(GPobj, TBobj)
%
% INPUTS
% GPobj         An object of the GenParams class, which defines dimensions,
%               matrix type, SNR, etc.  [Default: A default GenParams obj.]
% TBobj         An object of the TurboOpt class, which contains the
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
%
% OUTPUTS
% Y             M-by-T measurement matrix
% A             M-by-N random measurement matrix
% X             N-by-T true signal matrix
% S             N-by-T true support matrix
% Observation   An object of a concrete sub-class of the abstract
%               Observation class whose parameters define the *actual* 
%               statistics of the observation channel.  (This is needed if 
%               the SNR is defined in the GenParams object, and it differs 
%               from the parameters present in TurboOpt.Observation, as the 
%               SNR will override any TurboOpt.Observation properties)
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/30/13
% Change summary: 
%       - Created (01/03/12; JAZ)
%       - Added support for a time-varying transform matrix, A(t)
%         (02/07/12; JAZ)
%       - Renamed Noise class references to Observation class references
%         (05/23/12; JAZ)
%       - Changed how Observation object is copied (07/10/12; JAZ)
%       - Modified the Observation class-related genRand method calls
%         (01/30/13; JAZ)
% Version 0.2
%

function [Y, A, X, S, Observation] = signal_gen_fxn(GPobj, TBobj)

%% Create a default realization in the absence of any user inputs

if nargin == 0
    GPobj = GenParams();
    TBobj = TurboOpt();
elseif nargin == 1
    TBobj = TurboOpt();
end


%% Check inputs for errors

if ~isa(GPobj, 'GenParams')
    error('GenParams must be a valid GenParams object')
end
if ~isa(TBobj, 'TurboOpt')
    error('TurboOpt must be a valid TurboOpt object')
end


%% Construct a realization

% Extract problem sizes
N = GPobj.N;
T = GPobj.T;
M = GPobj.M;

% Create a signal realization
SigObj = TBobj.Signal;
[X, S] = SigObj.genRand(TBobj, GPobj);

% Create measurement matrix/matrices realization
A = cell(1,T);
for t = 1:T
    switch GPobj.A_type
        case 1,             % iid gaussian
            if strcmp(GPobj.type, 'complex')     % Complex-valued case
                A{t} = (randn(M,N) + 1j*randn(M,N))/sqrt(2*M);
                for n=1:N, A{t}(:,n) = A{t}(:,n)/norm(A{t}(:,n)); end;
            else
                A{t} = randn(M,N)/sqrt(M);
                for n=1:N, A{t}(:,n) = A{t}(:,n)/norm(A{t}(:,n)); end;
            end
        case 2,             % rademacher
            A{t} = sign(randn(M,N))/sqrt(M);     
        case 3,             % subsampled DFT
            if strcmp(GPobj.type, 'real'), 
                error('Inappropriate A_type for real-valued data'); 
            end
            mm = zeros(N,1); while sum(mm)<M, mm(ceil(rand(1)*N))=1; end; 
            A{t} = dftmtx(N); A{t} = A{t}(mm==1,:)/sqrt(M);
    end
end
switch GPobj.commonA
    case true, A = A{1};    % Reduce to a single matrix
    case false, % Nothing to do here
end

% Calculate transform coefficients
if GPobj.commonA
    Z = A*X;
else
    Z = cell(1,T);
    for t = 1:T
        Z{t} = A{t}*X(:,t);
    end
end

if isempty(GPobj.SNRdB)
    % Generate according to TBobj.Observation object properties
    ObservationObj = TBobj.Observation;
    
    Y = ObservationObj.genRand(TBobj, GPobj, Z);
    
    % Create an independent copy of the provided Observation object
    Observation = ObservationObj.copy();
else
    % Ignore the properties in the TBobj.Observation object.  Instead,
    % construct an object of the same class, but with parameters chosen to
    % yield the desired per-measurement SNR
    
    % Compute the signal power
    switch GPobj.commonA
        case true
            SigPwr = norm(Z, 'fro')^2;
        case false
            SigPwr = 0;
            for t = 1:T
                SigPwr = SigPwr + norm(Z{t})^2;
            end
    end
    
    % Create an independent copy of the provided Observation object
    Observation = TBobj.Observation.copy();
    
    switch class(TBobj.Observation)
        case 'GaussNoise'   % AWGN            
            % Set AWGN noise variance
            noisevar = (SigPwr / M / T) * 10^(-GPobj.SNRdB / 10);
            
            % Generate noise realization using this noise variance
            Observation.prior_var = noisevar;
            Y = Observation.genRand(TBobj, GPobj, Z);
            
        case 'GaussMixNoise'    % Binary Gaussian mixture
            % Use SNR (in dB) set by GenParams to get desired noise var.
            noisevar = (SigPwr / M / T) * 10^(-GPobj.SNRdB / 10);
            
%             % Now calculate the small (nu0) and large (nu1) variances that
%             % will, given mixing weights and ratio (nu1/nu0) yield the
%             % desired noise variance of noisevar
%             nu0 = noisevar ./ (1 + (Noise.NUratio - 1).*Noise.PI);
            nu0 = noisevar;
            
            % Generate noise realization using this noise variance
            Observation.NU0 = nu0;
            Y = Observation.genRand(TBobj, GPobj, Z);
            
        otherwise
            error(['I don''t know how to set the parameters of a %s ' ...
                'Observation object to yield the desired SNR'], ...
                class(TBobj.Observation))
    end
end
