function [xhat,pcoeff,plike] = gampEstPoly(cnt,A0,estimWt,iterOpt,gampOpt)

% Get dimensions
[nz,nx] = size(A0);

% Get options for overall estimation
npoly = iterOpt.npoly;  % number of terms in poly approx of nonlinearity
useLS = iterOpt.useLS;  % use LS fit for polynomial coefficients
pcoeffInit = iterOpt.pcoeffInit;    % Initial polynomial coefficient estiamte
nitTot = iterOpt.nitTot;            % number of iterations
verbose = iterOpt.verbose;          % print progress
np1 = npoly + 1;

% Initialize vectors
xhatTot = zeros(nx,nitTot);
pcoeffTot = zeros(np1,nitTot);
plike = zeros(nitTot,2);

% Get initial polynomial estimate of a linear relation
if (isempty(pcoeffInit))
    [xhat0,xvar0] = estimWt.estimInit();
    zmean = mean(A0*ones(nx,1))*xhat0;
    cntMean = mean(cnt);
    pcoeffInit = [cntMean/zmean 0]';
end
pcoeff = pcoeffInit;
polyMax = 0;

for it = 1:nitTot
    
    % Estimation of the linear weights
    % --------------------------------
    
    % Output distribution
    noiseVar = 0;
    rateMin = 1;
    outEstLin = NeuralOutPolyEst(cnt,noiseVar,pcoeff,polyMax,rateMin);   
    z0 = linspace(0,10,100)';
    rateEst = outEstLin.rateFn(z0);
    
    % Input distribution    
    z0mean = 0;
    z0var = 1;
    inEstLin = NeuralConnEstIn(nx, estimWt, z0mean, z0var);
    
    % Estimate polynomial coefficients
    Alin = [A0 ones(nz,1)];
    uhat = gampEst(inEstLin, outEstLin, Alin, gampOpt);
    xhat = uhat(1:nx);
    
    % Shift distribution    
    icutoff = round(nx*(1-2*estimWt.p1));
    xhatSort = sort(xhat);
    xcutoff = xhatSort(icutoff);
    bias = mean(xhatSort(1:icutoff));    
    xhat = (xhat-bias).*(xhat > xcutoff);
    xhatTot(:,it) = xhat;
    
    % Compute likelihood
    zhat = Alin*uhat;
    plike(it,1) = outEstLin.logLike(zhat);
    
    % Abort if likelihood did not increase
    if (verbose)
        fprintf(1,'iter=%d plike=%f\n', it, plike(it,1));
    end
    if (it > 1)
        if (plike(it,1) < plike(it-1,1))
            if (verbose)
                fprintf(1,'failed to increase\n');
            end
            xhat = xhatTot(:,it-1);
            break;
        end
    end
    
    % Estimation of the polynomial coefficients
    % -----------------------------------------
    zhat = A0*xhat;
    nz = length(zhat);
    Apoly = zeros(nz, np1);
    Apoly(:,np1) = ones(nz,1);
    for ip = npoly:-1:1
        Apoly(:,ip) = Apoly(:,ip+1).*zhat;
    end
    
    % Get initial estimate of polynomial
    if (cond(Apoly) > 1e6)
        disp('NeuralConnSim:  condition number');
        keyboard;
    end
    pcoeff0 = Apoly \ cnt;
    if (useLS)
        pcoeff = pcoeff0;
    else
        inEstPoly = AwgnEstimIn(pcoeff0, 1000*ones(np1,1));
        
        % Output estimator
        noiseVar = 0;
        pcoeff1 = [1 0]';
        rateMin = 1;
        outEstPoly = NeuralOutPolyEst(cnt,noiseVar,pcoeff1,0,rateMin);
        
        % Estimate polynomial coefficients      
        pcoeff = gampEst(inEstPoly, outEstPoly, Apoly, gampOpt);
       
    end
    pcoeffTot(:,it) = pcoeff;
    
    % Set maximum of polynomial interpolation to 90% largest value for z
    pz = 0.9;
    zsort = sort(zhat);
    polyMax = zsort(round(nz*pz));
    
    % Compute likelihood
    zhat0 = Apoly*pcoeff;
    plike(it,2) = outEstLin.logLike(zhat0);
    
end
return

% Plot result
z0 = linspace(0,10,100)';
rate = outFn.getTranserFn(z0);

v = polyval(pcoeff,z0);
rateEst = log(1 + exp(v));
plot(z0, [rate rateEst]);

r0 = log(1 + exp(polyval(pcoeff0,zhat)));
r1 = log(1 + exp(polyval(pcoeff,zhat)));
J0 = sum(cnt.*log(r0) - r0);
J1 = sum(cnt.*log(r1) - r1);


