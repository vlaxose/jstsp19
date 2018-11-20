% Options for PBiGAMPiidBGawgn.m
classdef PBiGAMPiidBGawgnOpt
    
    properties

        % reporting
        verbose = true; % verbose text output?
        plotFig = 10; % figure number of plot, or =0 for no plot

        % priors and likelihood
        cmplx = false; % complex valued signal and noise?
        meanB = 0; % mean of Gaussian in Bernoulli-Gaussian prior
        meanC = 0; % mean of Gaussian in Bernoulli-Gaussian prior
        varB = 1; % variance of Gaussian in Bernoulli-Gaussian prior
        varC = 1; % variance of Gaussian in Bernoulli-Gaussian prior
        sparB = 0.1; % sparsity of Bernoulli-Gaussian prior
        sparC = 0.1; % sparsity of Bernoulli-Gaussian prior
        wvar = 1e-10; % noise variance
        EM = false; % use EM to learn prior parameters and noise variance?

        % PBiGAMP options
        maxIt = 250; % maximum number of iterations per try
        stepInit = 0.3; % initial value of damping parameter [0.3]
        stepIncr = 1.006; % per-iter increase of damping parameter [1.006]
        stepMax = 0.5; % max damping parameter [0.5]
        tol = 1e-7; % stopping tolerance [1e-7]
        normTol = 1e-10; % if norm(zhat)<normTol, then exit [1e-10]

        % random re-starts
        maxTry = 20; % number of random re-tries 
        errTry = 10^(-60/10); % normalized residual error to stop trying
        varGainInit = 1; % gain on variance initialization [1]
        meanGainInitB = 1; % gain on mean initialization of B [1]
        meanGainInitC = 1; % gain on mean initialization of C [1]
        meanInitTypeB = 'randn'; % in {'randn','spike'} ['randn']
        meanInitTypeC = 'randn'; % in {'randn','spike'} ['randn']

        % ground truth
        btrue = []; % setting this will build errfxnB
        ctrue = []; % setting this will build errfxnC
        errfxnB = @(bhat) inf; % error function for b estimate
        errfxnC = @(chat) inf; % error function for c estimate 

    end % properties
    

    methods
        
        % Constructor with default options
        function opt = PBiGAMPiidBGawgnOpt(varargin)
            if nargin == 0
                % No custom parameters values, thus create default object
                return
            elseif mod(nargin, 2) == 0
                % User is providing property/value pairs
                names = fieldnames(opt);    % Get names of class properties
                
                % Iterate through every property/value pair, assigning
                % user-specified values.  Note that the matching is NOT
                % case-sensitive
                for i = 1 : 2 : nargin - 1
                    if any(strcmpi(varargin{i}, names))
                        % User has specified a valid property
                        propName = names(strcmpi(varargin{i}, names));
                        opt.(propName{1}) = varargin{i+1};
                    else
                        % Unrecognized property name
                        error('PBiGAMPiidBGawgnOpt: %s is an unrecognized option', ...
                            num2str(varargin{i}));
                    end
                end
                return
            else
                error(['PBiGAMPiidBGawgnOpt requires arguments ' ...
                    'be provided in pairs, e.g., PBiGAMPiidBGawgnOpt(''maxTry'',' ...
                    ' 20, ''EM'', false)'])
            end
        end

        % *****************************************************************
        %                           SET METHODS
        % *****************************************************************
        
        function obj = set.btrue(obj,btrue)
            obj.btrue = btrue;
            obj.errfxnB = @(bhat) (norm(bhat*((bhat'*btrue)/norm(bhat)^2)-btrue ...
                )/norm(btrue))^2;
        end

        function obj = set.ctrue(obj,ctrue)
            obj.ctrue = ctrue;
            obj.errfxnC = @(chat) (norm(chat*((chat'*ctrue)/norm(chat)^2)-ctrue ...
                )/norm(ctrue))^2;
        end

    end % methods
    
end
