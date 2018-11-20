function gampShowHist(estHist,opt,xTrue,zTrue,varargin)
%
% Graphically show per-iteration historical data created by gampEst
% 
% gampShowHist(estHist,optFin[,xTrue[,zTrue]])
% 
% e.g.
%       cd {gampmatlab}/examples/sparseEstim/
%       addpath .
%       sparseAWGN
%       opt.legacyOut = false;
%       opt.step = .1;  % slow motion 
%       opt.stepIncr = 1.1;
%       [estFin,optFin,estHist] = gampEst(inputEst, outputEst, A, opt);
%       gampShowHist(estHist,optFin,x)

%  Mark Borgerding (2014-03-03) borgerding dot 7 at osu edu

if nargin<2
    error('need two input arguments')
end
if nargin<3 
    xTrue=[];
end
if nargin<4 
    zTrue=[]; 
end

% set defaults
autoScale = false; % scale xhat to match norm of xTrue
if nargin>4 % parse input to change defaults
    validProps = {'autoScale'};
    for i = 1:2:length(varargin)
        validatestring(varargin{i},validProps);
        evalc([varargin{i},'=',num2str(varargin{i+1})]);
    end
end

[nx,nt] = size( estHist.xhat);
[nz,nt] = size( estHist.Axhat);
t=estHist.it;
nt=length(t);

stepSize = estHist.step;
stepMax = estHist.stepMax;

val = estHist.val;
pass = estHist.pass;
val(~isfinite(val))=nan;

if opt.removeMean 
    if ~isempty(xTrue)
        [nxTrue,nc] = size(xTrue);
        A = LinTransDemeanRC(MatrixLinTrans(randn(1,nxTrue))); % used to create A.contract operator
        estHistXhat = estHist.xhat; estHist.xhat = zeros(nxTrue*nc,nt);
        estHistRhat = estHist.rhat; estHist.rhat = zeros(nxTrue*nc,nt);
        for k=1:nt; % remove augmented entries from in history
            estHist.xhat(:,k) = reshape(A.contract(reshape(estHistXhat(:,k),nx/nc,nc)),nxTrue*nc,1);
            estHist.rhat(:,k) = reshape(A.contract(reshape(estHistRhat(:,k),nx/nc,nc)),nxTrue*nc,1);
        end
        nx = nxTrue*nc;
    end
    if ~isempty(zTrue)
        [nzTrue,nc] = size(zTrue);
        A = LinTransDemeanRC(MatrixLinTrans(randn(nzTrue,1))); % used to create A.contract operator
        estHistPhat = estHist.phat; estHist.phat = zeros(nzTrue*nc,nt);
        for k=1:nt; % remove augmented entries from in history
            estHist.phat(:,k) = reshape(A.contract(reshape(estHistPhat(:,k),nz/nc,nc)),nzTrue*nc,1);
        end
        nz = nzTrue*nc;
    end
end

if ~isempty(xTrue)
    if ~autoScale
        MSEx = arrayfun( @(k) sum( abs( xTrue(:) - estHist.xhat(:,k) ).^2 )/nx , 1:nt )';
        MSEr = arrayfun( @(k) sum( abs( xTrue(:) - estHist.rhat(:,k) ).^2 )/nx , 1:nt )';
    else
        normxTrue = norm(xTrue(:));
        MSEx = arrayfun( @(k) sum( abs( xTrue(:) - estHist.xhat(:,k)*(normxTrue/norm(estHist.xhat(:,k))) ).^2 )/nx , 1:nt )';
        MSEr = arrayfun( @(k) sum( abs( xTrue(:) - estHist.rhat(:,k)*(normxTrue/norm(estHist.rhat(:,k))) ).^2 )/nx , 1:nt )';
    end
else
    MSEx=[];
end
xvar = mean(estHist.xvar)';
rvar = mean(estHist.rvar)';

subplot(221)
plot( t, stepSize,'b.-',...
      t, stepMax,'m:',...
      t, opt.stepMin*ones(size(t)),'g:');
legend('step','stepMax','stepMin','Location','Best')
xlabel('iteration')
ylabel('stepsize')
grid on
title('Step vs. Iteration')

subplot(222)
semilogy(t, rvar,'c-',...
         t, xvar,'g-')
legs={'Rvar','Xvar'};
hh = ishold; hold on; 
if ~isempty(xTrue)
    semilogy(t, MSEr,'b-')
    semilogy(t, MSEx,'m-')
    legs={legs{:},'MSE(Rhat)','MSE(Xhat)'};
end
if ~hh, hold off; end
xlabel('iteration')
ylabel('MSE')
legend(legs{:},'Location','Best')
grid on
title('Variances vs. Iteration')

subplot(223)
plot( t, val,'b.-');
if all(estHist.pass)
    legend('pass','Location','Best')
else
    hh = ishold; hold on; 
    plot(t(find(~estHist.pass)),estHist.val(~estHist.pass),'r.'); 
    if ~hh, hold off; end
    legend('pass','fail','Location','Best')
end
xlabel('iteration')
ylabel('value')
grid on
title('Value vs. Iteration')

if ~isempty(xTrue)
    subplot(224)
    eXR = xTrue(:)*ones(1,nt) - estHist.rhat;
    cXR = sum(((conj(xTrue(:))/norm(xTrue(:)))*ones(1,nt)).*eXR,1)./sqrt(sum(abs(eXR).^2,1));
%cXR = corr(xTrue(:),(:)*ones(1,nt) - estHist.rhat);
    if any(~isreal(cXR))
        cXR = abs(cXR);
    end
    plot( t,cXR,'b-');
    if ~isempty(zTrue)
        eZP = zTrue(:)*ones(1,nt) - estHist.phat;
        cZP = sum(conj(estHist.phat).*eZP,1)./sqrt(sum(abs(eZP).^2,1).*sum(abs(estHist.phat).^2,1));
%cZP = corr(zTrue(:),zTrue(:)*ones(1,nt) - estHist.phat);
        %cZS = corr(zTrue(:),estHist.shat);
        if any(~isreal(cZP)) %|| any(~isreal(cZS))
            cZP = abs(cZP);
            %cZS = abs(cZS);
        end
        hh = ishold; hold on; 
        plot( t,cZP,'m-');
        %plot( t,cZS,'g-');
        if ~hh, hold off; end
%       legend( 'corr(Xtrue,Xtrue - Rhat )', 'corr(Ztrue,Ztrue - Phat )','corr(Ztrue,Shat)','Location','Best')
        legend( 'corr(Xtrue,Xtrue - Rhat )', 'corr(Phat,Ztrue - Phat )','Location','Best')
    else
        legend('corr(Xtrue,Xtrue - Rhat )','Location','Best')
    end
    xlabel('iteration')
    ylabel('correlation coefficient')
    grid on
    title('Correlation vs. Iteration')
end


