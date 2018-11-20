% This "inverts" the function obj.estim(phat;pvar) in that
% it solves for the value of phat that yields
%   obj.estim(phat;pvar) = Axhat
% for a specified Axhat.
%
% More generally, it solves % for the value of phat that yields
%   alpha*obj.estim(phat;pvar) + (1-alpha)*phat = Axhat
% for a specified Axhat and alpha, where alpha=1 is the default.

function [phat,zhat,zvar,gamma] = estimInvert(obj,Axhat,pvar,opt)

if (nargin<4)||(~isfield(opt,'debug')), opt.debug = false; end
if (nargin<4)||(~isfield(opt,'phat0')), opt.phat0 = Axhat; end
if (nargin<4)||(~isfield(opt,'alg')), opt.alg = 1; end
if (nargin<4)||(~isfield(opt,'maxIter')), opt.maxIter = 50; end;
if (nargin<4)||(~isfield(opt,'tol')), opt.tol = 1e-4; end;
if (nargin<4)||(~isfield(opt,'stepsize')), opt.stepsize = 0.5; end;
if (nargin<4)||(~isfield(opt,'regularization')), opt.regularization = 1e-20; end;
if (nargin<4)||(~isfield(opt,'alpha')), opt.alpha = 1; end;
if (nargin<4)||(~isfield(opt,'p')), opt.p = 2; end;

if opt.debug,
    phat_ = nan(length(Axhat(:)),opt.maxIter);
    zhat_ = nan(length(Axhat(:)),opt.maxIter);
end;

% Test quality of initialization
p = opt.p; % use ell_p norms throughout
alpha = opt.alpha;
phat0 = opt.phat0;
[zhat0,zvar0] = obj.estim(phat0,pvar);

%Handle the case of alpha ~= 1
if any(alpha ~= 1)
    alphaFlag = 1;
    zhatHolder = zhat0;
else
    alphaFlag = 0;
end

%Continue
zhat0 = alpha.*zhat0 + (1-alpha).*phat0;
residual0 = Axhat-zhat0;
NR0 = norm(residual0(:),p);



% Iteratively improve estimate of fixed-point
gamma = opt.stepsize; % initial stepsize
maxIter = opt.maxIter;
for t = 1:10 % repeated tries at smaller stepsizes (if needed)
    
    % initialize
    NR = NR0;
    NRold = inf;
    phat = phat0;
    zhat = zhat0;
    zvar = zvar0;
    residual = residual0;
    phatOld = inf(size(phat));
    r = 0;
    
    % iterative root-finding
    while (r<maxIter) ...
            &&(NR>opt.tol*norm(zhat(:),p)) ...
            &&(abs(NR-NRold)>opt.tol*NR)
        
        r = r+1;
        if opt.debug
            phat_(:,r) = phat(:);
            zhat_(:,r) = zhat(:);
        end
        
        % Update estimate
        phatOld = phat;
        NRold = NR;
        switch opt.alg
            case 0 % simple method (very slow!)
                phat = phatOld + gamma*residual;
            case 1 % approximation of halley's method
                gradient = alpha.*(zvar./pvar) + (1-alpha);
                phat = phatOld + gamma*residual.*gradient./(gradient.^2 ...
                    + opt.regularization);
            case 2 % newton's method for root-finding, with regularization
                gradient = alpha.*(zvar./pvar) + (1-alpha);
                phat = phatOld + gamma*residual./(gradient ...
                    + opt.regularization);
        end
        
        % Test quality
        [zhat,zvar] = obj.estim(phat,pvar);
        if alphaFlag
            zhatHolder = zhat; %save the true zhat
            zhat = alpha.*zhat + (1-alpha).*phat;
        end
        residual = Axhat-zhat;
        NR = norm(residual(:),p);
        
    end % while
    
    % Present debug information
    if opt.debug
        % report iterations and error
        fprintf(1,'estimInvert iterations=%3g\n',r);
        fprintf(1,'norm(zhat-Axhat,%i)=%5.5g\n',...
            p,norm(zhat(:)-Axhat(:),p)) %#ok<*PRTCAL>
        fprintf(1,'norm(zhat-Axhat,%i)/norm(zhat,%i)=%5.5g\n',...
            p,p,norm(zhat(:)-Axhat(:),p)/norm(zhat(:),p))
        fprintf(1,'norm(phat-phatOld,%i)/norm(phat,%i)=%5.5g\n\n',...
            p,p,norm(phat(:)-phatOld(:),p)/norm(phat(:),p))
        
        % plot example trajectories
        figure(100); clf;
        subplot(211)
        plot(1:r,real(phat_(:,1:r))')
        grid on
        ylabel('real(phat)')
        xlabel('iteration')
        title('estimInvert coefficient trajectories')
        subplot(212)
        Axhat_ = Axhat(:);
        semilogy(1:r,abs(zhat_(:,1:r)-Axhat_(:)*ones(1,r))')
        grid on
        ylabel('abs(zhat-Axhat)')
        xlabel('iteration')
        drawnow
        pause
    end % opt.debug
    
    % Reduce stepsize if no progress was made
    if (NR>1.01*NR0)
        gamma = gamma/5;  % decrease stepsize
        %[t,NR/NR0]
        %maxIter = maxIter*5  % increase iterations
    else
        break; % successful estimation of fixed-point
    end
    
end %t

%Reset zhat to match the estimator output (only matter when alpha ~= 1)
if alphaFlag
    zhat = zhatHolder;
end

% warn if no progress was made even after reducing stepsize
if (NR>1.01*NR0)
    warning(['estimInvert: No progress made after ',num2str(t),...
        ' attempts with different stepsizes: NR0=',num2str(NR0), ...
        ', NR=',num2str(NR),'. Adjust estimInvert options!'])
end

