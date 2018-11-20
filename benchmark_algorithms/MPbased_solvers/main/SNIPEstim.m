classdef SNIPEstim < EstimIn
    % SNIPE:  Sparsifying Non-Informative Parameter Estimator
    %
    % Author Mark Borgerding (borgerding.7 osu edu)
    %
    % Parameters:
    % isCmplx: logical: true when input signal is complex-valued
    % omega: real number: typically in range -4 to 10 
    %        (qualitatively, larger omega implies more shrinkage)
    % varEps: small non-negative number
    %
    % SNIPE is the MMSE estimator of X from observation X + Normal(0,rvar).
    % where X is distributed according to the pdf
    %     p_X(x) = (1-lambda) Normal(x;0,varEps) + lambda*p_0(x/sigma)/sigma
    % for an _arbitrary_ p_0() that is continuous at the origin,
    % in the limit of sigma->infty and lambda->1 such that sigma*(1-lambda) 
    % converges to a positive number (to avoid a trivial estimator).
    %
    % When isCmplx=false, 
    %   Normal() is the standard normal pdf and
    %   sigma*(lambda-1) -> exp(omega)/sqrt(2*pi*rvar)/p_0(0)
    % and when isCmplx=true, 
    %   Normal() is a circular-Gaussian pdf and
    %   sigma*(lambda-1) -> exp(omega)/sqrt(pi*rvar)/p_0(0).
    % 
    % Note: when varEps=0, Normal(x;0,varEps) reduces to the Dirac delta.
    %
    % Reference:
    % Generalized Approximate Message Passing for the Cosparse Analysis Model
    % Mark Borgerding, Phil Schniter http://arxiv.org/abs/1312.3968
    properties 
        omega=[];
        xhat0=0;
        xvar0=1; % only used when SNIPE is used as an EstimIn, as a return value from estimInit
        xvarEps=0; % the variance of the small values
        xvarBig = inf;  % allows one to consider a very large finite variance (in Bernoulli-Gaussian sense)
        isCmplx=false; % true for complex-valued inputs
    end

    methods
        % Constructor
        function obj = SNIPEstim(omega,varargin)
            obj = obj@EstimIn;
            if nargin>0, obj.omega = omega; end
            for i = 1:2:length(varargin)
                obj.(varargin{i}) = varargin{i+1};
            end
        end

        function [xhat,xvar,val] = estim(obj, rhat, rvar)

            if (obj.isCmplx)

                if obj.xvarEps<=0
                    rho = 1 ./ (1 + exp(-abs(rhat).^2 ./ rvar + obj.omega ) );
                    xhat = rho .* rhat;  
                    % xvar := rvar times the xhat derivative w.r.t. rhat
                    xvar = rho .* ( abs(rhat).^2 .* (1-rho) + rvar);
                else
                    e = exp(-abs(rhat).^2 ./ rvar + obj.omega );
                    de_dr = e .* (-2*abs(rhat)./rvar);
                    vr = 1./(1+rvar/obj.xvarEps);
                    u = (1+vr.*e);
                    du_dr = vr.*de_dr;
                    v = (1+e);
                    dv_dr = de_dr;
                    g = u./v;
                    dg_dr = ( du_dr.*v - u.*dv_dr )./ v.^2;
                    xhat = rhat.*g;
                    dx_dr = abs(rhat.*dg_dr) + g;
                    xvar = rvar.*dx_dr;
                end

            else

                if any(~isreal(rhat))
                    warning('SNIPE is ignoring imaginary part of input. If undesired, set isCmplx=true.');
                end
      
                if obj.xvarEps<=0
                    rho = 1 ./ (1 + exp(-.5*real(rhat).^2 ./ rvar + obj.omega ) );
                    xhat = rho .* rhat;  
                    % xvar := rvar times the xhat derivative w.r.t. rhat
                    xvar = rho .* ( real(rhat).^2 .* (1-rho) + rvar);
                else
                    e = exp(-.5*real(rhat).^2 ./ rvar + obj.omega );
                    de_dr = e .* (-rhat./rvar);
                    vr = 1./(1+rvar/obj.xvarEps);
                    u = (1+vr.*e);
                    du_dr = vr.*de_dr;
                    v = (1+e);
                    dv_dr = de_dr;
                    g = u./v;
                    dg_dr = ( du_dr.*v - u.*dv_dr )./ v.^2;
                    xhat = rhat.*g;
                    dx_dr = rhat.*dg_dr + g;
                    xvar = rvar.*dx_dr;
                end

            end

            % consider a large finite variance to get some convergence guarantees
            if isfinite( obj.xvarBig )
                gain = 1./(1 + rvar./obj.xvarBig );
                xhat = xhat .* gain;
                xvar = xvar .* gain;
            end

            if nargout < 3
                val=0;
            else
                % The details of derivation of negative KL divergence are in kldivinput.tex
                if obj.isCmplx
                    tmp = abs(rhat).^2 ./ rvar;
                    tmpMax = max(tmp,obj.omega);
                    tmpMin = min(tmp,obj.omega);
                    val = - tmpMin + log( 1 + exp(tmpMin - tmpMax) )  + ( xvar + abs(xhat-rhat).^2) ./rvar;
                else
                    tmp = 0.5*real(rhat).^2 ./ rvar;
                    tmpMax = max(tmp,obj.omega);
                    tmpMin = min(tmp,obj.omega);
                    val = - tmpMin + log( 1 + exp(tmpMin - tmpMax) )  + ( xvar + abs(xhat-rhat).^2) ./rvar / 2;
                end
            end
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
        function ll = logScale(obj,Axhat,pvar,phat)

            % Find the fixed-point of phat
            opt.phat0 = Axhat; 
            opt.alg = 1; % approximate newton's method
            opt.maxIter = 50;
            opt.tol = 1e-4;
            opt.stepsize = 0.5;
            opt.regularization = 0;  
            opt.debug = false;
            phatfix = estimInvert(obj,Axhat,pvar,opt);

            if ~obj.isCmplx
                % Real case
                % Compute log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar)
                % = log(1-lambda)               
                %       - log(2*pi*pvar)/2 
                %       + log( exp(-.5*phatfix^2./pvar) + exp(-omega) )

                ls1 = 0;  % ignoring "log(1-lambda)" term as a constant
                if isfinite(obj.xvarBig) % unless we have been told how to find it
                    ls1 = -log(1+exp( .5*log(obj.xvarBig./pvar)  - obj.omega));
                end
                ls2 = -.5*log( 2 * pi * pvar );

                % log( exp(-.5*phatfix^2./pvar) + exp(-omega) )
                % = -tmp + log(1 + exp(tmp-obj.omega)) 
                % = -obj.omega + log(1 + exp(obj.omega-tmp))
                tmp = 0.5*real(phatfix).^2 ./ pvar;
                tmpMax = max(tmp,obj.omega);
                tmpMin = min(tmp,obj.omega);
                ls3 = -tmpMin + log( 1 + exp(tmpMin - tmpMax) ); % note : log(exp(-a)+exp(-b)) = -a + log(1+exp(a-b))

                % Combine with the upper bound on the entropy to form output cost
                Hgauss = 0.5*(real(Axhat - phatfix)).^2./pvar;

                ll = ls1 + ls2 + ls3 + Hgauss;
            else
                % similar for complex
                % log int_z p_{Y|Z}(y|z) CN(z;phatfix, pvar)
                % = log(1-lambda)      
                %       - log(pi*pvar)
                %       + log( exp(-abs(phatfix)^2./pvar) + exp(-omega) )

                ls1 = 0;    % ignoring "log(1-lambda)" term as a constant
                if isfinite(obj.xvarBig) % unless we have been told how to find it
                    ls1 = -log(1+exp( log(obj.xvarBig./pvar)  - obj.omega));
                end
                ls2 = -log( pi * pvar ); 

                tmp = abs(phatfix).^2 ./ pvar;
                tmpMax = max(tmp,obj.omega);
                tmpMin = min(tmp,obj.omega);
                ls3 = -tmpMin + log( 1 + exp(tmpMin - tmpMax) ); % log( exp(-abs(phatfix)^2./pvar) + exp(-omega) )

                Hgauss =  (abs(Axhat - phatfix)).^2./pvar;  % entropy term differs
                ll = ls1 + ls2 + ls3 + Hgauss;
            end
        end

        function [xhat0,xvar0,val0] = estimInit(obj)
            xhat0 = obj.xhat0;
            xvar0 = obj.xvar0;
            val0 = 0;
        end

        function [dgdo,d2go2] = sdDeriv(obj,rhat,rvar,omega)
            % Squared Distance Derivative
            % let f(rhat,rvar,omega) := norm(rhat - xhat)^2
            % where xhat(rhat,rvar,omega) := SNIPEstim(omega).estim(rhat,rvar)
            % then this function returns the first and second partial derivative of
            % f(.) w.r.t the tuning parameter omega
            if nargin<4
                omega = obj.omega;
            end
            h = exp(.5*abs(rhat).^2./rvar - omega);
            dgdo  = 2*sum( abs(rhat).^2 .* h ./ (1+h).^3);
            d2go2 = 2*sum( abs(rhat).^2 .* h .* (2*h-1) ./ (1+h).^4);
        end

    end
    
    methods (Static)
        function om = omegaSolve(rhat,rvar,targetSD)
            % returns the omega value such that
            % norm(rhat - SNIPEstim(omega).estim(rhat,rvar) )^2 = targetSD
            % In other words, 
            % minimize g(omega) = ( f(omega) - targetSD)^2 /2
            % where f(omega) = norm(rhat - xhat_omega)^2
            om = 0;
            rh2 = abs(rhat).^2;
            erh2v = exp( abs(rhat).^2/2./rvar );

            g = @(om) (sum( rh2./ (1 +  exp(-om)*erh2v).^2 ) - targetSD)^2/2;

            %fprintf('rvar=%g targetSD=%g (perco=%g) frac=%g',mean(rvar),targetSD,targetSD/length(rhat),targetSD/norm(rhat)^2 )
            scaleMax=1; sigma = .1;beta = .1; % Armijo stepsize parameters
            gcur = g(om);
            for k=1:50
                h = exp(-om)*erh2v;
                scale2 = (1 + h).^-2;
                scale2(isinf(h))=0;
                scale3 = (1 + h).^-3;
                scale3(isinf(h))=0;
                y = sum( rh2.*scale2 ) - targetSD; % want this to equal 0
                perCoefD1 = 2*rh2 .* h .*scale3;
                perCoefD1(~isfinite(perCoefD1))=0;
                d1  =  y*sum(perCoefD1);
                d =  -d1; % steepest descent
                
                if abs(d) < 1e-7 
                    break
                end

                if isnan(d)
                    d
                end
                % Armijo stepsize rule, with growable max stepsize scaleMax
                m=0;
                while true
                    alphak = beta.^m*scaleMax;
                    g_alphak =  g(om+alphak*d);
                    if gcur - g_alphak >= -sigma*alphak*d1*d
                        break;
                    end
                    m = m + 1;
                    if m>20
                        break;
                    end
                end
                gcur = g_alphak;
                scaleMax = alphak/beta; % allow stepsize to grow to find those hard-to-reach places
                om = om + alphak*d;
                %fprintf('%d: om=%.3f alphak=%g d=%g sqerr=%g m=%d\n',k,om, alphak,d,g_alphak,m);
            end
        end
    end
end
