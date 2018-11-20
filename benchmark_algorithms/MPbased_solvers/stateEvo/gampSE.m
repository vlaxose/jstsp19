function [mseSE, hist] = gampSE(inAvg, outAvg, SEopt)
% gampSE:  State evolution analysis for GAMP
%
% The classes inAvg and outAvg provide methods for the SE updates at the
% input and output nodes depending on the distribution:
%
% [xcov0,taux0] = inAvg.seInit()
%    Initial true covariance on x and postulated MSE.
% [xcov,taux] = inAvg.seIn(xir, alphar, taur);
%   True covariance on x and posutlated MSE given input measurement
%   parameters.
% 
% [taur,xir,alphar] = outAvg.seOut(zpcov, taup)
%   Postulated output variance, true output variance and gain.

% Get options
beta = SEopt.beta;          % measurement ratio nx/nz
Avar = SEopt.Avar;          % normalized variance of A:  nz*var(A(i,j))
tauxieq = SEopt.tauxieq;    % Forces taur(t) = xir(t)
nit = SEopt.nit;            % number of iterations
verbose = SEopt.verbose;    % 1=display progress

% Initialize variables for history
taux = zeros(nit,1);
taup = zeros(nit-1,1);
taur = zeros(nit-1,1);
xir = zeros(nit-1,1);
alphar = zeros(nit-1,1);
xcov = zeros(2,2,nit);
zpcov = zeros(2,2,nit-1);

% Mean squared error on x in dB
mseSE = zeros(nit,1);

% SE initialization
[xcov0, taux0] = inAvg.seInit();
xvar0 = [1 -1]*xcov0*[1; -1];
xcov(:,:,1) = xcov0;
taux(1) = taux0;
mseSE(1) = 10*log10( [1 -1]*xcov0*[1; -1] / xvar0 );

% Main SE loop
for it = 1:nit-1
    
    % Compute zpcov = cov(Z,P) and taup = postulated variance on P
    zpcov(:,:,it) = Avar*beta*xcov(:,:,it);
    taup(it) = Avar*beta*taux(it);
   
    % Output update    
    [tauri,xiri,alphari] = outAvg.seOut(zpcov(:,:,it), taup(it));
    taur(it) = tauri/Avar;
    xir(it) = xiri/Avar;
    alphar(it) = alphari; 
    
    % Force taur = xir, if requested
    if (tauxieq)
        xir(it) = taur(it);
    end
    
    % Input update
    [xcovi,tauxi] = inAvg.seIn(xir(it), alphar(it), taur(it));    
    taux(it+1) = tauxi;
    xcov(:,:,it+1) = xcovi;   
    mseSE(it+1) = 10*log10( [1 -1]*xcovi*[1; -1] / xvar0 );
    
    % Print progress, if requested
    if (verbose)
        fprintf(1,'it=%d MSE=%12.4e\n', it, mseSE(it+1));
    end
end

% Store results in history structure
hist.taux = taux;
hist.taup = taup;
hist.taur = taur;
hist.xir = xir;
hist.alphar = alphar;
hist.xcov = xcov;
hist.zpcov = zpcov;

