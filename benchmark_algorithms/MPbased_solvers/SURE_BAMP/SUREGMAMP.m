
%
function [Xhat, SUREfin, estHist, optSUREfin, optGAMPfin] = SUREGMAMP(Y, A, optSURE, optGAMP)

% If A is an explicit matrix, replace by an operator
if isa(A, 'double')
    A = MatrixLinTrans(A);
end

%Find problem dimensions
[M,N] = A.size();
T = size(Y, 2);
if size(Y, 1) ~= M
    error('Dimension mismatch betweeen Y and A')
end

%Merge user-specified GAMP and SURE options with defaults
if nargin <= 2
    optSURE = [];
end
if nargin <= 3
    optGAMP = [];
end
[optGAMP, optSURE] = check_opts(optGAMP, optSURE);
optSURE.SUREtol = min(10.^(-optSURE.SNRdB/10-1),optSURE.maxTol);

%Initialize GM parameters
if optSURE.L == 1
    %Initialize BG
    [lambda, theta, phi, optSURE] = set_initsBG(optSURE, Y, A, M, N, T);
    L = 1;
    if optSURE.heavy_tailed
        theta = zeros(N,T);
        optSURE.learn_mean = false;
    end
else
    %Initialize GM
    [lambda, omega, theta, phi, optSURE] = set_initsGM(optSURE, Y, A, M, N, T);
    L = size(theta,3);
    if (L ~= size(phi,3) || L ~= size(omega,3))
        error('There are an unequal amount of components for the active means, variances, and weights')
    end
end

%Handle history saving/reporting
histFlag = false;
if nargout >=3
    histFlag = true;
    estHist = [];
end;

%Initialize loop
muw = optSURE.noise_var;
firstStep = optGAMP.step;
t = 0;
stop = 0;

%Initialize XhatPrev
XhatPrev = inf(N,T);

while stop == 0
    %Increment time exit loop if exceeds maximum time
    t = t + 1;
    if t >= optSURE.maxEMiter
        stop = 1;
    end
    
    %Input channel for real or complex signal distributions
    if ~optSURE.cmplx_in
        if L == 1
            inputEst = AwgnEstimIn(theta, phi);
        else
            inputEst = GMEstimIn(omega,theta,phi);
        end
    else
        if L == 1
            inputEst = CAwgnEstimIn(theta, phi);
        else
            inputEst = CGMEstimIn(omega,theta,phi);
        end
    end
    inputEst = SparseScaEstim(inputEst,lambda);

    %Output channel for real or complex noise distributions
    if ~optSURE.cmplx_out
        outputEst = AwgnEstimOut(Y, muw);
    else
        outputEst = CAwgnEstimOut(Y, muw);
    end

    %Perform GAMP
    if ~histFlag
        estFin = gampEst(inputEst, outputEst, A, optGAMP);
    else
        [estFin,~,estHistNew] = gampEst(inputEst, outputEst, A, optGAMP);
        estHist = appendEstHist(estHist,estHistNew);
    end

    Xhat = estFin.xhat;
    Xvar = estFin.xvar;
    Rhat = estFin.rhat;
    %If rhats are returned as NaN, then gamp failed to return a better
    %estimate,  SURE has nothing to go on, so break.
    if any(isnan(Rhat(:))); break; end;
    Rvar = estFin.rvar;
    Shat = estFin.shat;
    Svar = estFin.svar;
    %If doing SURE learning of noise with hidden Z use these variables
    if optSURE.hiddenZ
        Zhat = estFin.zhat;
        Zvar = estFin.zvar;
    %If doing SURE learning of noise with hidden X use these variables 
    else
        Zhat = A.mult(Xhat);
        Zvar = A.multSq(Xvar);
    end

    %Update parameters for either real or complex signal distributions
    if ~optSURE.cmplx_in
        Rhat = real(Rhat);
        %Update BG parameters
        [lambda, theta, phi, SUREfin] = CBG_update(Rhat, Rvar, lambda, theta, phi, optSURE);
 
    end

    phi = max(phi,optSURE.minVar);

    %Update noise variance. Include only a portion of the Zvar
    %in beginning stages of SUREGMAMP because true update may make it
    %unstable.  Same update for real or complex noise.
    if optSURE.learn_noisevar
        if ~optSURE.maxBethe
            if strcmp(optSURE.noise_dim,'joint')
                muw = sum(sum(abs(Y-Zhat).^2))/M/T + sum(sum(Zvar))/M/T ;
                SNRdB = 10*log10( norm(Y,'fro')./muw );
            elseif strcmp(optSURE.noise_dim,'col')
                muw = sum(abs(Y-Zhat).^2,1)/M +sum(Zvar,1)/M ;
                SNRdB = 10*log10( sum(abs(Y.^2))./muw );
            elseif strcmp(optSURE.noise_dim,'row')
                muw = sum(abs(Y-Zhat).^2,2)/T+sum(Zvar,2)/T;
                SNRdB = 10*log10( sum(abs(Y.^2),2)./muw );
            end
        else
            if strcmp(optSURE.noise_dim,'joint')
                muw = muw.*sum(sum(abs(Shat).^2))./sum(sum(Svar));
                SNRdB = 10*log10( norm(Y,'fro')./muw );
            elseif strcmp(optSURE.noise_dim,'col')
                 muw = muw(1,:).*sum(abs(Shat).^2)./sum(Svar);
                SNRdB = 10*log10( sum(abs(Y.^2))./muw );
            elseif strcmp(optSURE.noise_dim,'row')
                muw = muw(:,1).*sum(abs(Shat).^2,2)./sum(Svar,2);
                SNRdB = 10*log10( sum(abs(Y.^2),2)./muw );
            end
        end
    else
        SNRdB = 10*log10( norm(Y,'fro')./muw(1,1) );
    end
    optSURE.SUREtol = min(min(10.^(-SNRdB/10-1)),optSURE.maxTol);
    
    muw = resize(muw,M,T);

    %Calculate the change in signal estimates
    norm_change = norm(Xhat-XhatPrev,'fro')^2/norm(Xhat,'fro')^2;

    %Check for estimate tolerance threshold
    if norm_change < optSURE.SUREtol
        stop = 1;
    end

    %Warm-start reinitialization of GAMP
    XhatPrev = Xhat;
    optGAMP = optGAMP.warmStart(estFin);
%     optGAMP.xhat0 = Xhat;
%     optGAMP.xvar0 = Xvar;
%     optGAMP.shat0 = estFin.shat;
%     optGAMP.svar0 = estFin.svar;
%     optGAMP.xhatPrev0 = estFin.xhatPrev;
%     optGAMP.scaleFac = estFin.scaleFac;
%     optGAMP.step = min(max(estFin.step,firstStep),estFin.stepMax);
%     %optGAMP.step = estFin.step;
%     optGAMP.stepMax = estFin.stepMax;

end;


%Do a final FULL SURE update of noise var (psi)
if optSURE.learn_noisevar
    if ~optSURE.maxBethe
        if strcmp(optSURE.noise_dim,'joint')
            muw = sum(sum(abs(Y-Zhat).^2))/M/T+sum(sum(Zvar))/M/T;
        elseif strcmp(optSURE.noise_dim,'col')
            muw = sum(abs(Y-Zhat).^2,1)/M+sum(Zvar,1)/M;
        elseif strcmp(optSURE.noise_dim,'row')
            muw = sum(abs(Y-Zhat).^2,2)/T+sum(Zvar,2)/T;
        end
    else
        if strcmp(optSURE.noise_dim,'joint')
            muw = muw.*sum(sum(abs(Shat).^2))./sum(sum(Svar));
        elseif strcmp(optSURE.noise_dim,'col')
            muw = muw(1,:).*sum(abs(Shat).^2)./sum(Svar);
        elseif strcmp(optSURE.noise_dim,'row')
            muw = muw(:,1).*sum(abs(Shat).^2,2)./sum(Svar,2);
        end
    end
end
muw = resize(muw,M,T);

%Input channel for real or complex signal distributions
if ~optSURE.cmplx_in
    if L == 1
        inputEst = AwgnEstimIn(theta, phi);
    else
        inputEst = GMEstimIn(omega,theta,phi);
    end
else
    if L == 1
        inputEst = CAwgnEstimIn(theta, phi);
    else
        inputEst = CGMEstimIn(omega,theta,phi);
    end
end
inputEst = SparseScaEstim(inputEst,lambda);

%Output channel for real or complex noise distributions
if ~optSURE.cmplx_out
    outputEst = AwgnEstimOut(Y, muw);
else
    outputEst = CAwgnEstimOut(Y, muw);
end

%Perform GAMP
if ~histFlag
    estFin = gampEst(inputEst, outputEst, A, optGAMP);
else
    [estFin,~,estHistNew] = gampEst(inputEst, outputEst, A, optGAMP);
    estHist = appendEstHist(estHist,estHistNew);
end

%Output final solution 
Xhat = estFin.xhat;

%Output final parameter estimates
SUREfin.Xvar = estFin.xvar;
SUREfin.Zhat = estFin.zhat;
SUREfin.Zvar = estFin.zvar;
SUREfin.Rhat = estFin.rhat;
SUREfin.Rvar = estFin.rvar;
SUREfin.Phat = estFin.phat;
SUREfin.Pvar = estFin.pvar;

%BG parameters
if L ==1
    if strcmp(optSURE.sig_dim,'joint')
        SUREfin.lambda = lambda(1,1);
        SUREfin.active_mean = theta(1,1);
        SUREfin.active_var = phi(1,1);
    elseif strcmp(optSURE.sig_dim,'col')
        SUREfin.lambda = lambda(1,:);
        SUREfin.active_mean = theta(1,:);
        SUREfin.active_var = phi(1,:);
    elseif strcmp(optSURE.sig_dim,'row')
        SUREfin.lambda = lambda(:,1);
        SUREfin.active_mean = theta(:,1);
        SUREfin.active_var = phi(:,1);
    end
    
%GM parameters
else
    if strcmp(optSURE.sig_dim,'joint')
        SUREfin.lambda = lambda(1,1);
        SUREfin.active_weights = reshape(omega(1,1,:),1,L)';
        SUREfin.active_mean = reshape(theta(1,1,:),1,L).';
        SUREfin.active_var = reshape(phi(1,1,:),1,L)';
    elseif strcmp(optSURE.sig_dim,'col')
        SUREfin.lambda = lambda(1,:);
        SUREfin.active_weights = reshape(omega(1,:,:),T,L)';
        SUREfin.active_mean = reshape(theta(1,:,:),T,L).';
        SUREfin.active_var = reshape(phi(1,:,:),T,L)';
    elseif strcmp(optSURE.sig_dim,'row')
        SUREfin.lambda = lambda(:,1);
        SUREfin.active_weights = reshape(omega(:,1,:),N,L)';
        SUREfin.active_mean = reshape(theta(:,1,:),N,L).';
        SUREfin.active_var = reshape(phi(:,1,:),N,L)';
    end
end

if strcmp(optSURE.noise_dim,'joint')
    SUREfin.noise_var = muw(1,1);
elseif strcmp(optSURE.noise_dim,'col')
    SUREfin.noise_var = muw(1,:);
elseif strcmp(optSURE.noise_dim,'row')
    SUREfin.noise_var = muw(:,1);
end

%Output final options
optSURE = rmfield(optSURE,'noise_var');
optSUREfin = optSURE;
optGAMPfin = optGAMP;


return;
