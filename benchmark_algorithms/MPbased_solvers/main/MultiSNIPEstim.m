classdef MultiSNIPEstim < EstimIn
    % MultiSNIPE:  Sparsifying Non-Informative Parameter Estimator
    % With multiple points of discrete probability (delta functions in pdf)
    %
    % Author Mark Borgerding (borgerding.7 osu edu)
    %
    % Parameters:
    % theta: vector of dirac delta positions
    % omega: scalar or vector of real "gravity" parameter(s): typically in range -4 to 10 
    %        (qualitatively, larger omega implies more shrinkage)
    % xvarBig: Provide magnitude shrinkage no less than an AWGN Estimator with xvarBig variance
    %
    % SNIPE is the MMSE estimator of X from observation X + Normal(0,rvar).
    % where X is distributed according to the pdf
    %     p_X(x) = sum_{i=1:L} ( lambda_i delta(x-theta_i) ) + lambda_0*p_0(x/sigma)/sigma
    % for an _arbitrary_ p_0() that is continuous at the origin,
    % in the limit of sigma->infty and lambda->1 such that sigma*(1-lambda) 
    % converges to a positive number (to avoid a trivial estimator).
    %
    % Reference:
    % Generalized Approximate Message Passing for the Cosparse Analysis Model
    % Mark Borgerding, Phil Schniter http://arxiv.org/abs/1312.3968
    properties 
        theta=[];
        omega=[];
        xhat0=0;
        xvar0=1; % only used when SNIPE is used as an EstimIn, as a return value from estimInit
        xvarBig = inf;  % allows one to consider a very large finite variance (in Bernoulli-Gaussian sense)
    end

    methods
        % Constructor
        function obj = MultiSNIPEstim(theta,omega,varargin)
            obj = obj@EstimIn;
            if nargin>0, obj.theta = theta(:).'; end
            if nargin>1, obj.omega = omega(:).'; end
            for i = 1:2:length(varargin)
                obj.(varargin{i}) = varargin{i+1};
            end
        end

        function [xhat,xvar,val] = estim(obj, rhat, rvar)
            nr = length(rhat);
            if length(rvar) == 1, rvar = rvar*ones(size(rhat));end
            L = length(obj.theta);
            if length(obj.omega)==1,obj.omega = obj.omega * ones(size(obj.theta));end
            dterm = abs(rhat * ones(1,L) - ones(nr,1)*obj.theta).^2 ./ ( rvar * ones(1,L) );
            eterm = exp( ones(nr,1)*obj.omega - dterm/2);

            d0 = sum(eterm,2)+1;
            d1 = eterm * obj.theta' + rhat;
            d2 = eterm * (obj.theta').^2 + rhat.^2 + rvar;
            xhat = d1./d0;
            xvar = d2./d0 - xhat.^2;
            logScale = log(d0);

            % consider a large finite variance to get some convergence guarantees
            if isfinite( obj.xvarBig )
                gain = 1./(1 + rvar./obj.xvarBig );
                xhat = xhat .* gain;
                xvar = xvar .* gain;
            end
            val = logScale + .5*(log(2*pi*rvar) + (xhat-rhat).^2./rvar + xvar./rvar);
        end

        % Compute old cost when used as an output estimator:
        function todo = logLike(obj,xhat,xvar)
            todo = zeros(size(xhat)); 
            error('Use opt.adaptStepBethe=true with SNIPE!');
        end

        % Compute Bethe cost when used as an output estimator:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        %    The first part is Hgauss (entropy term)
        %
        % derivation follows that of log( C_j ) in kldivinput.tex 
        % except for the entropy term Hgauss
        function [ll,phatfix] = logScale(obj,Axhat,pvar,phat)

            % Find the fixed-point of phat
            opt.phat0 = Axhat; 
            opt.alg = 1; % approximate newton's method
            opt.maxIter = 50;
            opt.tol = 1e-4;
            opt.stepsize = 0.5;
            opt.regularization = 0;  
            opt.debug = false;
            phatfix = estimInvert(obj,Axhat,pvar,opt);

            % similar for complex
            % log int_z p_{Y|Z}(y|z) CN(z;phatfix, pvar)
            % = log(1-lambda)      
            %       - log(pi*pvar)
            %       + log( exp(-abs(phatfix)^2./pvar) + exp(-omega) )

            ls1 = 0;    % ignoring "log(1-lambda)" term as a constant
            if isfinite(obj.xvarBig) % unless we have been told how to find it
                ls1 = -log(1+exp( log(obj.xvarBig./pvar)  - obj.omega));
            end

            nr = length(phatfix);
            if length(pvar) == 1, pvar = pvar*ones(size(phatfix));end
            L = length(obj.theta);
            if length(obj.omega)==1,obj.omega = obj.omega * ones(size(obj.theta));end
            dterm = abs(phatfix * ones(1,L) - ones(nr,1)*obj.theta).^2 ./ ( pvar * ones(1,L) );
            eterm = exp( ones(nr,1)*obj.omega - dterm/2);

            d0 = sum(eterm,2)+1;
            ls2 = log(d0);

            Hgauss = 0.5*(real(Axhat - phatfix)).^2./pvar;
            ll = ls1 + ls2 + Hgauss;
        end

        function [xhat0,xvar0,val0] = estimInit(obj)
            xhat0 = obj.xhat0;
            xvar0 = obj.xvar0;
            val0 = 0;
        end
    end
end
