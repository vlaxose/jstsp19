%EM-GM-AMP-MOS  EM-GM-AMP Model Order Selection
% From the (possibly complex-valued) M-by-T matrix of noisy linear 
% observations Y = AX + W, EMGMAMP returns approximately MMSE estimates 
% of the N-by-T signal matrix X, where the focus is the sub-Nyquist case 
% M < N, as in sparse reconstruction and compressive sensing.  In EMGMAMP,
% it is assumed that the signal coefficients X(n,t) are apriori independent 
% and Bernoulli-Gaussian-mixture distributed according to the pdf:
%
%   p(X(n,t)) = (1-lambda(t))*delta(X(n,t))
%      + sum_k^L lambda(t)*omega(t,k) (C)Normal(X(n,t);theta(t,k),phi(t,k))
%
% where n = 1:N indexes the row of the signal matrix X,  t = 1:T 
% indexes the column (timestep) of all matrices, and delta() denotes the
% Dirac delta pdf.  Above, lambda(t), omega(t,:), theta(t,:), and phi(t,:) 
% are the sparsity rate, mixture weights, active-coefficient means, and 
% active-coefficient variances, respectively, of the coefficients within 
% the t-th column of X.  In EMGMAMP, the noise samples W(m,t) are assumed 
% apriori independent and zero-mean Gaussian distributed: 
%
%    p(W(m,t)) = (C)Normal(W(m,t);0,psi(t))
%
% where m = 1:M indexes the rows of the matrices W and Y, and psi(t) 
% is the variance within the t-th column of W.
%
% The estimate of X is calculated using the Generalized Approximate Message 
% Passing (GAMP) algorithm, as described in "Generalized Approximate Message 
% Passing for Estimation with Random Linear Mixing" by Sundeep Rangan,
% with certain adaptive-stepsize enhancements.
%
% The parameters lambda, omega, theta, phi, and psi are automatically 
% learned via the Expectation Maximization (EM) algorithm, using X and W as 
% the "hidden" data. 
%
% The EMGMAMPMOS algorithm is described in detail in "Expectation-Maximization
% Gaussian-Mixture Approximate Message Passing" by Jeremy Vila and 
% Philip Schniter.
% 
% EM-GM-AMP-MOS calls EM-GM-AMP as a subroutine, and estimates a lower
% bound on the likelihood function given the estimated parameters.  Then,
% using a Bayesian Information Criterion (BIC) penalty term, it
% automatically estimates a new model order.  This is iterated until
% subsequent model orders are the same, or until maximum iterations are
% reached.
%
%Syntax:
% [Xhat, EMfin] = EMGMAMPMOS(Y, A) % suggested default 
% [Xhat, EMfin, estHist, optEMfin] = EMGMAMPMOS(Y, A, optEM, optGAMP) % full version
%
%Inputs:
% Y - matrix of observed measurements
%
% A - known mixing matrix (in either explicit or GAMP-object format)
%
% optEM - structure containing various EM option fields [optional]
%   .heavy_tailed       Set to true for heavy-tailed (compressible or
%                       non-compressible) signals.  [default=true]
%   .SNRdB              dB SNR used to initialize noise variance [default=20]
%   .learn_lambda       Set to true to learn lambda, set to false
%                       to never update lambda (stays at initialization)
%                       [default=true]
%   .learn_weights      Set to true to learn the weights omega of the
%                       Gaussian Mixture [default=true]
%   .learn_mean         Set to true to learn active mean (theta), set to false
%                       to never update theta (stays at initialization)
%                       [default=true]
%   .learn_var          Set to true to learn active variance (phi), set to
%                       false to never update active variance (stays at
%                       initialization) [default=true]
%   .learn_noisevar     Set to true to learn noise variance (psi), set to
%                       false to never update noise variance (stays at
%                       initialization) [default=true]
%   .maxBethe           Set to true to learn the output parameters via
%                       maximization of the Bethe Free Energy [default =
%                       true]
%   .sig_dim            Set to 'col' to allow different EM-learned signal
%                       params for each column in matrix X.  Set to 'row' 
%                       to allow different EM-learned signal params for 
%                       each row in matrix X. Set to 'joint' to force 
%                       common EM-learned parameters for all columns.
%                       [default='col']
%   .noise_dim          Set to 'col' to allow different EM-learned noise
%                       params for each column in matrix Y.  Set to 'row' 
%                       to allow different EM-learned signal params for 
%                       each row in matrix Y. Set to 'joint' to force 
%                       common EM-learned parameters for all columns.
%                       [default='joint']
%   .robust_gamp        Toggle on when the sensing matrix gives issues 
%                       (e.g., correlated columns). This option 
%                       automatically overrides GAMP options to handle 
%                       these cases [default = false]
%   .Lmax               Maximum number of components L.   (default = 5)
%   .maxLiter           Maixmum number iterations to estimate L. 
%                       (default = 3)
%
%   Note: EMGMAMPMOS(Y, A) or EMGMAMPMOS(Y, A, []) sets EM params at defaults
%
%   WARNING! Set the following initializations only if confident they
%   accurately represent the model.  Otherwise, let the defaults preside.
%   Also, ensure that all GM-component vectors are of compatible lengths.  
%
%   .L                  Number of mixture components [default=3, max=20]
%   .cmplx_in           Set to true if input is complex.  [default = 
%                       ~isreal(A^T*randn(M,1)) || ~isreal(Y)]
%   .cmplx_out          Set to true if output is complex.  [default = 
%                       ~isreal(Y)]
%   .lambda             Initialize overall sparsity rate lambda. Must be 
%                       either scalar or 1-by-T vector.  [default is 
%                       based on the LASSO phase transition curve]
%   .active_weights     Initialize GM component weights (omega).
%                       Must be L-by-1 or L-by-T vector.
%                       [Defaults based on best L-term uniform-pdf fit]
%   .active_mean        Initialize mean of active components (theta)
%                       Must be L-by-1 or L-by-T vector.
%                       [Defaults based on best L-term uniform-pdf fit]
%   .active_var         Initialize variance of active components (phi)
%                       Must be L-by-1 or L-by-T vector.
%                       [Defaults based on best L-term uniform-pdf fit]
%   .noise_var          Initialize the noise variance.  Must be either 
%                       scalar or 1-by-T vector [default is based on .SNRdB]
%   
% optGAMP - structure containing various GAMP option fields [optional] (see GampOpt.m)
%
%   Note: EMGMAMPMOS(Y, A) or EMGMAMPMOS(Y, A, [], []) sets GAMP parameters at defaults
%
%   Note: Setting optEM.heavy_tailed=true overrides the above parameter 
%   initializations, and sets optEM.learn_mean=false.
%
%Outputs:
% Xhat - GAMP-estimated posterior means
%
% EMfin - Contains EM-estiated signal and noise parameters.
%   .Xvar               GAMP-estimated MMSE variance
%   .Rhat               GAMP quantity
%   .Rvar               GAMP quantity
%   .Zhat               GAMP-estimated MMSE mean of noiseless measurement
%   .Zvar               GAMP-estimated MMSE var of noiseless measurement
%   .Phat               GAMP quantity
%   .Pvar               GAMP quantity
%   .pi_n               Posterior activity probabilities.
%   .gamma_n            Posterior mean.
%   .nu_n               Posterior variance.
%   .beta_n             Per-Component posterior scaling quantities.
%   .lambda             Sparsity rate (lambda) [1-by-T for 'col' mode]
%   .active_weights     weights of GM (omega) [L-by-T for 'col' mode]
%   .active_mean        Means of active components (theta) [L-by-T for 'col' mode]
%   .active_var         Variances of active components (phi) [L-by-T for 'col' mode]
%   .noise_var          Variance of noise (psi) [1-by-T for 'col' mode]
%   .BIC                BIC cost matrix.  Row indexes j, column indexes
%                       hypothesized L from 1 until cost decreases
%
% estHist - GAMP output structure containing per-iteration GAMP data (see gampEst.m)
%
% optEMfin - Contains the final values of optEM
%
% optGAMPfin - Contains the final values of optGAMP
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 6/14/13
% Change summary: 
%   v 1.0 (JV)- First release.  NOTE! NOT TESTED FOR MMV MODEL.
%   v 1.2 (JV)- Enhanced for numerical stability.  NOTE! NOT TESTED FOR MMV MODEL.
%   v 4.2 (JV)- Adjusted EM and GAMP defaults for enhanced spped and
%               robustness.  Also adjusted input/ouput format.
%   v 4.3 (JV)- optEM.heavy_tailed = true is now the default.  Also added
%               optEM.robust_gamp option to handle cases when matrix is 
%               problematic
%   v 4.4 (JV)- added optEM.sig_dim = 'row' and optEM.noise_dim = 'row' to
%               learn EM parameters on each row.
%   v 6.0 (JV)- changed noise-variance learning method, and ditched the 
%               external EM loop.  Now learning is done inside the EstimIn
%               and EstimOut functions
% Version 6.0
%

function [xhat, EMfin, estHist, optEMfin] = EMGMAMPMOS(y, A, optEM, optGAMP)

% If A is an explicit matrix, replace by an operator
if isa(A, 'double')
    A = MatrixLinTrans(A);
end

%Find problem dimensions
[M,N] = A.size();
if size(y, 1) ~= M
    error('Dimension mismatch betweeen Y and A')
end
T = size(y,2);

%Set default GAMP options if unspecified
if nargin <= 2
    optEM = [];
end

if nargin <= 3
    optGAMP = [];
end

%Check inputs and initializations
[optGAMP, optEM] = check_opts(optGAMP, optEM);

L = optEM.L;

histFlag = false;

if nargout >=3
    histFlag = true;
    estHist.step = [];
    estHist.stepMax = [];
    estHist.val = [];
    estHist.pass = [];
end;

if isfield(optEM,'cmplx_in')
    cmplx = optEM.cmplx_in;
else
    if ~isreal(A.multTr(randn(M,1))) || ~isreal(y)
        cmplx = true;
    else
        cmplx = false;
    end
end

%Get scaling factors for different learning dimensions
if strcmp(optEM.sig_dim,'joint')
    dimConst = 1;
elseif strcmp(optEM.sig_dim,'row')
    dimConst = N;
elseif strcmp(optEM.sig_dim,'col')
    dimConst = T;
end

%Perform EMGMAMP with L0
if histFlag
    [xhat, EMfin, estHist, optEMfin] = EMGMAMP(y, A, optEM, optGAMP);
else
    [xhat, EMfin] = EMGMAMP(y, A, optEM, optGAMP);
end
Rhat = EMfin.Rhat; Rvar = EMfin.Rvar;

ind = isnan(Rhat); Rhat(ind) = xhat(ind);

%set iteration count
t = 0;
stop = 0;
BIC = nan(optEM.maxLiter,optEM.Lmax);
while stop == 0
    t = t + 1;
    
    %if model order iterations go above a threshold stop
    if t > optEM.maxLiter
       break
    end
      
    if L == 1
        EMfin.active_weights = 1;
    end
    
    if strcmp(optEM.sig_dim,'col')
        phi = resize(permute(EMfin.active_var,[3,2,1]),N,T,L);
        theta = resize(permute(EMfin.active_mean,[3,2,1]),N,T,L);
        omega = resize(permute(EMfin.active_weights,[3,2,1]),N,T,L);
    else
        phi = resize(permute(EMfin.active_var,[2,3,1]),N,T,L);
        theta = resize(permute(EMfin.active_mean,[2,3,1]),N,T,L);
        omega = resize(permute(EMfin.active_weights,[2,3,1]),N,T,L);
    end
    lambda = resize(EMfin.lambda,N,T);
    
    %Preallocate storage
    gamma = zeros(N,T,L); alpha = zeros(N,T,L);
    beta = zeros(N,T,L); nu = zeros(N,T,L);
    a_n = zeros(N,T,L);

    %Get posterior quantities
    if cmplx
        for i = 1:L
           beta(:,:,i) = phi(:,:,i) + Rvar + eps;
           alpha(:,:,i) = abs(Rhat-theta(:,:,i)).^2./beta(:,:,i);
           gamma(:,:,i) = (Rhat.*phi(:,:,i) + theta(:,:,i).*Rvar)./beta(:,:,i);
           nu(:,:,i) = Rvar.*phi(:,:,i)./beta(:,:,i);
           a_n(:,:,i) = Rvar./(beta(:,:,i)).*omega(:,:,i)...
            .*exp((abs(Rhat-theta(:,:,i)).^2./beta(:,:,i)-abs(Rhat).^2./Rvar)./(-1));
        end

        lik = zeros(N,T,L);
        for i = 1:L
            lik = lik + repmat(omega(:,:,i),[1 1 L])./omega...
                .*beta./repmat(beta(:,:,i),[1 1 L])...
                .*exp((alpha-repmat(alpha(:,:,i),[1 1 L])));
        end
        
    else
        for i = 1:L
           beta(:,:,i) = phi(:,:,i) + Rvar + eps;
           alpha(:,:,i) = (Rhat-theta(:,:,i)).^2./beta(:,:,i);
           gamma(:,:,i) = (Rhat.*phi(:,:,i) + theta(:,:,i).*Rvar)./beta(:,:,i);
           nu(:,:,i) = Rvar.*phi(:,:,i)./beta(:,:,i);
           a_n(:,:,i) = sqrt(Rvar./(beta(:,:,i))).*omega(:,:,i)...
            .*exp((abs(Rhat-theta(:,:,i)).^2./beta(:,:,i)-abs(Rhat).^2./Rvar)./(-2));
        end

        lik = zeros(N,T,L);
        for i = 1:L
            lik = lik + repmat(omega(:,:,i),[1 1 L])./omega...
                .*sqrt(beta./repmat(beta(:,:,i),[1 1 L]))...
                .*exp((alpha-repmat(alpha(:,:,i),[1 1 L]))/2);
        end
    end

    %Find posterior that the component x(n,t) is active
    pi_n = lambda./(1-lambda).*sum(a_n,3);
    pi_n = 1./(1+pi_n.^(-1));
    pi_n(isnan(pi_n)) = 0.001;
    
    %Learn the ML estimates of the parameters via EM loops for each
    %hypothesized model order, and evaluate cost function
    for i = 1:optEM.Lmax
        optEM.L = i;

        %initialize EM learning given a new number of components
        if i == 1
            %Initialize BG
            [lambda, theta, phi, ~] = set_initsBG(optEM, y, A, M, N, T);
            omega = 1;
        else
            %Initialize GM
            [lambda, omega, theta, phi, ~] = set_initsGM(optEM, y, A, M, N, T);
        end

        %Learn parameters based on posterior of L^0
        for iter =1:optEM.maxEMiter
            if cmplx
                if i ==1
                    %Update BG parameters
                    [lambda, theta, phi,~] = CBG_update(Rhat, Rvar, lambda, theta, phi, optEM);
                else
                    %Update GM parameters
                    [lambda, omega, theta, phi, ~] = CGM_update(Rhat, Rvar, lambda, omega, theta, phi, optEM);
                end
            else
                if i ==1
                    %Update BG parameters
                    [lambda, theta, phi,~] = BG_update(Rhat, Rvar, lambda, theta, phi, optEM);
                else
                    %Update GM parameters
                    [lambda, omega, theta, phi, ~] = GM_update(Rhat, Rvar, lambda, omega, theta, phi, optEM);
                end
            end
        end
        
        tmpSum = 0;
        for l = 1:L
            %reshape each gamma so that it fit dims of hypothesis model
            tmpGamma = repmat(gamma(:,:,l),[1,1,i]);
            %Evaluate the lower bound of the likelihood given 
            %p(x|y) = \delta(x- \gamma_\ell)
            if cmplx
                sumprior = sum(omega.*exp(-abs(tmpGamma-theta).^2./phi)/pi./phi,3);
            else
                sumprior = sum(omega.*exp(-(tmpGamma-theta).^2/2./phi)./sqrt(2*pi*phi),3);
            end
            tmpSum = log(sumprior)./lik(:,:,l) + tmpSum;
        end
        
        tmpSum(isinf(tmpSum)) = 0;
        postlik = sum(sum(pi_n.*tmpSum));
       
        %Compute cardinality of parameters given the various options
        if optEM.heavy_tailed
            mo = 2*i-1;
        else
            if cmplx
                mo = 4*i -1;
            else
                mo = 3*i -1;
            end
        end
        %Scale model order depending on how signal prior is defined
        %(joint,row,col)
        mo = mo*dimConst;
        BIC(t,i) = real(postlik) - mo*log(sum(pi_n(:)));
        
        %Check if BIC penality decreased, if so break from loop.
        if i > 1 && (BIC(t,i) < BIC(t,i-1))
            break
        end
    end
    
    %Find minimum BIC
    [~,Lnew] = max(BIC(t,:));
    
    %If previous L equals new L, then break.  There is no need to run the
    %same algorithm again.
    if L == Lnew
        stop = 1;
        break
    end
    L = Lnew; optEM.L = L;
    
    %Run again now with new L
    if histFlag
        [xhat, EMfin, estHist, optEMfin] = EMGMAMP(y, A, optEM, optGAMP);
    else
        [xhat, EMfin] = EMGMAMP(y, A, optEM, optGAMP);
    end
    Rhat = EMfin.Rhat; Rvar = EMfin.Rvar;
    
end

%Report final BIC value
EMfin.BIC = BIC;

return
