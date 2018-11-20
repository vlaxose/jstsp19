function [ut,vt,hist] =  rankOneFit(A,estimu,estimv,wvar,opt,u0,v0)
% rankOneFit:  Finds a rank one fit to a matrix
%
% Given a matrix A, the program attempts to fit a rank one model
% of the matrix
%
%   A = u0*v0'+ sqrt(m*wvar)*randn(m,n)
%
% The outputs (ut,vt) are the estimates of (u0,v0).  The structure hist
% contains a number of variables per iteration for debugging.
%
% The parameters estimu and estimv are the estimators for the u0 and v0
% which depend on their proability distribution.  They should be derived
% from the EstimIn class.  
%
% The parameter opt is a class of options, see RankOneOpt.
%
% For debugging, the subroutine can also accept true vectors u0 and v0.

% Get dimensions
[m,n] = size(A);
beta = n/m;

% Options
nit = opt.nit;  % number of iterations

% Check if true values are available for debugging
if (nargin < 7)
    opt.compTrue = 0;
end
if (~opt.compTrue)
    opt.pgenie = 0;
    opt.qgenie = 0;
    opt.SEgenie = 0;
end

% Indices to store value
indTheory = 1;  % theoretical values
indTrue = 2;    % true values
if (opt.compTrue)
    nval=2;   
else
    nval=1;
end

% Get second-order statistics of u0 and v0
[umean0, uvar0] = estimu.estimInit();
[vmean0, vvar0] = estimv.estimInit();
vsq0 = vmean0^2+vvar0;
usq0 = umean0^2+uvar0;
if (opt.compTrue)
    vsq = [vsq0 norm(v0)^2/n];
    usq = [usq0 norm(u0)^2/m];
else
    vsq = vsq0;
    usq = usq0;
end

% Correlations and their estimates
% av0 = E(vt^2) av1 = E(v0*vt) 
av0 = zeros(nit,nval);
av1 = zeros(nit,nval);
au0 = zeros(nit,nval);
au1 = zeros(nit,nval);
corru = zeros(nit,nval);
corrv = zeros(nit+1,nval);
if (opt.traceHist)
    hist.ut = zeros(m,nit);
    hist.vt = zeros(n,nit+1);
else
    hist = [];
end
hist.muv = zeros(nit,2);

% Initialize v to mean
vt = repmat(vmean0,n,1) + sqrt(opt.vvarInit)*randn(n,1);
ut = zeros(m,1);

% Compute initial second order statistics for v
av0(1,indTheory) = vmean0^2;
av1(1,indTheory) = vmean0^2;
if (opt.compTrue)
    av1(1,indTrue) = v0'*vt/n;
    av0(1,indTrue) = vt'*vt/n;
end
if (opt.traceHist)
    hist.vt(:,1) = vt;
end
muu = 0;
corrv(1,:) = abs(av1(1,:)).^2./av0(1,:)./vsq;

% Select index for second-order SE updates based on whether genie is
% supported
if (opt.SEgenie)
    ise=indTrue;
else
    ise=indTheory;
end

for it = 1:nit
    
    % U iteration
    % -----------
    % Linear step:  Computes
    %   p = (n/m)*av1(it,indTrue)*u0 + (1/m)*W*vt + muu*ut;
    p = (1/m)*A*vt + muu*ut;
    
    if (opt.linEst)
        % Linear estimation
        scale = sqrt(m)/norm(p);
        ut = scale*p;
        muv = -wvar*scale;
        au0(it,indTheory) = 1;
        corru(it,indTheory) = beta*usq0*vsq0*corrv(it,indTheory)/...
            (beta*usq0*vsq0*corrv(it,indTheory)+wvar);
        au1(it,indTheory) = sqrt(corru(it,indTheory)*usq0);
    else
        % General estimation based on observation
        %   p = pscale*u0 + z, z = N(0, pvar)
        pvar = (n/m)*wvar*av0(it,ise);
        pscale = (n/m)*av1(it,ise);
        y = p/pscale;               % Rescale value
        yvar1 = pvar/pscale^2;
        yvar = repmat(yvar1,m,1);
        
        % Call estimator for U
        [ut, uvart] = estimu.estim(y,yvar);
        
        % Compute mean squared error
        uvart = max( mean(uvart), opt.minau*uvar0);
        uvart = min( uvar0*yvar1/(uvar0+yvar1), uvart );
        
        % Compute theoretical second-order statistics, normalize u
        % and damping factor
        au1(it,indTheory) = max( usq(indTheory) - uvart, ...
            opt.minau*usq(indTheory));
        au0(it,indTheory) = au1(it,indTheory);
        muv = -wvar*uvart/yvar1/pscale;
        corru(it,indTheory) = abs(au1(it,indTheory))^2/au0(it,indTheory)/usq0;
        
        % Normalize output to theoretical value if requested.
        % This appears to stabilize the iterations considerably
        if (opt.normu)
            scale = sqrt(m*au0(it,indTheory))/norm(ut);
            ut = scale*ut;
        end
    end
    
    % Compute true second order statistics for u
    if (opt.compTrue)
        au1(it,indTrue) = u0'*ut/m;
        au0(it,indTrue) = ut'*ut/m;
        corru(it,indTrue) = abs(au1(it,indTrue))^2/au0(it,indTrue)/...
            usq(indTrue);
    end
    
    % Save result for u
    if (opt.traceHist)
        hist.ut(:,it) = ut;
    end
    
    
    % V iteration
    % -----------
    
    % Linear step
    q = (1/m)*A'*ut + muv*vt;
    
    if (opt.linEst)
        % Linear estimation
        vt = q;
        muu = -(n/m)*wvar;
        corrv(it+1,indTheory) = usq0*vsq0*corru(it,indTheory)/...
            (usq0*vsq0*corru(it,indTheory)+wvar);
        av0(it+1,indTheory) = q'*q/n;
        av1(it+1,indTheory) = sqrt(av0(it+1,indTheory)*corrv(it+1,indTheory)*vsq0);
        
    else
        % General estimation 
        % Compute scale factors such taht
        %   q = qscale*v0 + (1/m)*W'*ut + muv*vt
        qscale = au1(it,ise);
        qvar = wvar*au0(it,ise);
        
        % Rescale and call v estimator
        y = q/qscale;
        yvar1 = qvar/qscale^2;
        yvar = repmat(yvar1,n,1);
        [vt, vvart] = estimv.estim(y,yvar);
        vvart = max( mean(vvart), opt.minav*vvar0);
        
        % Compute theoretical second-order statistics and normalize v
        av1(it+1,indTheory) = max(vsq(indTheory) - vvart, ...
            opt.minav*vsq(indTheory));
        av0(it+1,indTheory) = av1(it+1,indTheory);
        corrv(it+1,indTheory) = abs(av1(it+1,indTheory))^2/av0(it+1,indTheory)/vsq0;
        muu = -(n/m)*wvar*vvart/yvar1/qscale;
        
        % Normalize output to theoretical value if requested.
        % This appears to stabilize the iterations considerably
        if (opt.normv)
            scale = sqrt(n*av0(it+1,indTheory))/norm(vt);
            vt = scale*vt;
        end
    end
    
    % Second order statistics for u
    if (opt.compTrue)
        av1(it+1,indTrue) = v0'*vt/n;
        av0(it+1,indTrue) = norm(vt)^2/n;
        corrv(it+1,indTrue) = abs(av1(it+1,indTrue))^2/av0(it+1,indTrue)/...
            vsq(indTrue);
    end
    
    % Save result for v
    if (opt.traceHist)
        hist.vt(:,it+1) = vt;
    end
end

% Add second-order stats to history
if (opt.traceHist)
    hist.au0 = au0;
    hist.au1 = au1;
    hist.av0 = av0;
    hist.av1 = av1;
    hist.corru = corru;
    hist.corrv = corrv;
end

