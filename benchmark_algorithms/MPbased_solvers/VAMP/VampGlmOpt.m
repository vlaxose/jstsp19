classdef VampGlmOpt
    % Options class for VampGlmEstim.m

    properties
        nitMax = 50;    % maximum number of VAMP iterations
        tol = 1e-4;     % stopping tolerance
        gamMin = 1e-8;  % minimum allowed precision [1e-11]
        gamMax = 1e14;   % maximum allowed precision [1e11]
        damp = 0.9;     % damping parameter in (0,1]

        Ah = [];        % fxn handle for A', needed only if A is a fxn handle
        N = [];         % =size(A,2), needed only if A is a fxn handle

        U = [];         % matrix of eigenvectors of A*A', or fxn handle,
                        % used (if present) when M<=N
                        %   [U,D]=eig(A*A'); d=diag(D);
        V = [];         % matrix of eigenvectors of A'*A, or fxn handle,
                        % used (if present) when M>N
                        %   [V,D]=eig(A'*A); d=diag(D);
        d = [];         % vector of eigenvalues of A*A' or A'*A
        Uh = [];        % fxn handle for U', needed only if U is a fxn handle
        Vh = [];        % fxn handle for V', needed only if V is a fxn handle

        r1init = [];    % initial value of vector r1 (estimate of x)
        gam1xinit = 1e-8;  % initial value of scalar gam1x (precision on r1)
        p1init = [];    % initial value of vector p1 (estimate of z)
        gam1zinit = 1e-8;  % initial value of scalar gam1z (precision on p1)

        learnGam1 = false;% learn gam1 instead of using standard VAMP update?

        verbose = false;% verbose switch 
        fxnErr1 = [];   % handle to a fxn of x1,z1 for error reporting, e.g.,
                        %   fxnErr1 = @(x1,z1) 10*log10( ...
                        %                    sum(abs(x1-xTrue).^2,1) ...
                        %                    ./sum(abs(xTrue).^2,1) );
        fxnErr2 = [];   % handle to another fxn of x1,z1 for error reporting
        fxnStop = [];   % handle to a stopping fxn of the form
                        %   fxnStop(i,err1(:,i),err2(:,i),...
                        %           r1old,r1,gam1x,x1,eta1x,...
                        %           p1old,p1,gam1z,z1,eta1z,...
                        %           r2old,r2,gam2x,x2,eta2x,...
                        %           p2old,p2,gam2z,z2,eta2z);
        histIntvl = 1;  % can save memory by decimating the saved history

        divChange = 1e-3; % amount to perturb input for denoiser Monte-Carlo
                        % divergence estimate; passed to FxnhandleEstimIn 
                        % when the denoiser reports a single output 
    end

    methods
        
        % Constructor with default options
        function opt = VampGlmOpt(varargin)
            if nargin == 0
                % No custom parameters values, thus create default object
                return
            elseif mod(nargin, 2) == 0
                % User is providing property/value pairs
                names = fieldnames(opt);    % Get names of class properties

                % Iterate through every property/value pair, assigning
                % user-specified values.  Note that the matching is NOT
                % case-sensitive
                for i = 1:2:nargin-1
                    if any(strcmpi(varargin{i}, names))
                        % User has specified a valid property
                        propName = names(strcmpi(varargin{i}, names));
                        opt.(propName{1}) = varargin{i+1};
                    else
                        % Unrecognized property name
                        error('VampGlmOpt: %s is an unrecognized option', ...
                            num2str(varargin{i}));
                    end
                end
                return
            else
                error(['The VampGlmOpt constructor requires arguments ' ...
                    'be provided in pairs, e.g., VampGlmOpt(''verbose'',' ...
                    ' false, ''nitMax'', 50)'])
            end
        end % VampGlmOpt

    end % methods

end
