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
%   .maxEMiter          number of EM iterations [default=20]
%   .SNRdB              dB SNR used to initialize noise variance [default=20]
%   .EMtol              EMGMAMP terminates when norm change between 
%                       sparsity rate is less than this tolerance. Set to
%                       -1 to turn off and run max iters [default=1e-5]
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
%   .minVar             Set minimum GM-component variance [default 1e-5]
%   .robust_gamp        Toggle on when the sensing matrix gives issues 
%                       (e.g., correlated columns). This option 
%                       automatically overrides GAMP options to handle 
%                       these cases [default = false]
%   .Lmax               Maximum number of components L.   (default = 5)
%   .maxLiter           Maixmum number iterations to estimate L. 
%                       (default = 3)
%
%   Note: EMGMAMP(Y, A) or EMGMAMP(Y, A, []) sets EM params at defaults
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
% optGAMP - structure containing various GAMP option fields [optional].
%   .nit                number of iterations
%   .pvarMin            minimum variance of each element of p
%   .XvarMin            minimum variance of each element of x
%   .step               step size 
%   .stepMin            minimum step size 
%   .stepMax            maximum step size
%   .stepIncr           Multiplicative step size increase, when successful
%   .stepDecr           Multiplicative step size decrease, when unsuccessful
%   .pvarStep           Logical flag to include a step size in the pvar/zvar calculation.
%   .varNorm            Option to "normalize" variances for computation.
%   .adaptStep          adaptive step size [important if A column-correlated]
%   .stepWindow         step size check window size
%   .bbStep             Barzilai Borwein step size [if A column-correlated]
%   .verbose            Print results in each iteration
%   .tol                Convergence tolerance
%   .stepTol            minimum allowed step size
%   .Avar               variance in A entries (may be scalar or matrix)
%   .xhat0              the initialization of x for GAMP [default handled
%                       internally]
%   .xvar0              the intialization of variance of each x_n for GAMP
%                       [default handled internally]
%   .shat0              the initialization of s for GAMP [default handled
%                       internally]
%
%   Note: EMGMAMP(Y, A) or EMGMAMP(Y, A, [], []) sets GAMP parameters at defaults
%
%   Note: Setting optEM.heavy_tailed=true overrides the above parameter 
%   initializations, and sets optEM.learn_mean=false.
%
%Outputs:
% Xhat - GAMP-estimated posterior means
%
% EMfin               Contains various posterior quantities needed for EM
%                       updates of the signal and noise.
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
%   .BIC                BIC cost function for each choice of L
%                       [maxLiter-by-Lmax]
%
% estHist - GAMP output structure containing per-iteration GAMP data
%   .pass               Per iteration logical value indicating successful step 
%   .step               Per-iteration stepsizes.
%   .stepMax            Per-iteration maximum allowed stepsizes.
%   .val                Per-iteration value of the GAMP cost function.
% 
% optEMfin - Contains the final values of EMopt
%
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
%
% Version 4.4
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
if T > 1
    error('EMGMAMPMOS does not support the MMV problem')
end

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

%Perform EMGMAMP with L0
if histFlag
    [xhat, EMfin, estHist, optEMfin] = EMGMAMP(y, A, optEM, optGAMP);
else
    [xhat, EMfin] = EMGMAMP(y, A, optEM, optGAMP);
end
Rhat = EMfin.Rhat; Rvar = EMfin.Rvar;

ind = isnan(Rhat); Rhat(ind) = xhat(ind);

gamma = repmat(EMfin.gamma_n,[1,optEM.Lmax,1]); gamma = permute(gamma,[1,3,2]);
if L ==1
    beta = ones(N,1,optEM.Lmax);
end
if L ~= 1
    beta = permute(EMfin.beta_n,[1,3,2])./repmat(sum(EMfin.beta_n,3),[1 L]);
end

pi_n = EMfin.pi_n;

beta(isnan(beta)) = 1/L;

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
    
    for i = 1:optEM.Lmax
        optEM.L = i;

        %initialize EM learning given a new number of components
        if i == 1
            %Initialize BG
            [lambda, theta, phi, ~] = set_initsBG(optEM, y, A, M, N, 1);
            omega = 1;
        else
            %Initialize GM
            [lambda, omega, theta, phi, ~] = set_initsGM(optEM, y, A, M, N, 1);
        end

        %Learn parameters based on posterior of L^0
        for iter =1:optEM.maxEMiter
            if i ==1
                %Update BG parameters
                [lambda, theta, phi,~] = BG_update(Rhat, Rvar, lambda, theta, phi, optEM);
            else
                %Update GM parameters
                [lambda, omega, theta, phi, ~] = GM_update(Rhat, Rvar, lambda, omega, theta, phi, optEM);
            end
        end
        
        %Reshape the parameters to 1:i vector (i = number of prior parameters) 
        param2.active_weights = reshape(omega(1,:,:),1,i)';
        param2.active_mean = reshape(theta(1,:,:),1,i).';
        param2.active_var = reshape(phi(1,:,:),1,i)';
        param2.lambda = lambda(1);
        
        if L ~= 1
            theta = repmat(theta,[1,L,1]);
            phi = repmat(phi,[1,L,1]);
            if i ~= 1
                omega = repmat(omega,[1,L,1]);
            end
        end
        
        %Evaluate the lower bound of the likelihood given 
        %p(x|y) = \delta(x- \gamma_\ell)
        if cmplx
            temp2 = sum(omega.*exp(-abs(gamma(:,:,1:i)-theta).^2./phi)/pi./phi,3);
        else
            temp2 = sum(omega.*exp(-(gamma(:,:,1:i)-theta).^2/2./phi)./sqrt(2*pi*phi),3);
        end
        temp = log(temp2);
        %At any point where this point is -Inf, set to something very
        %small.
        %temp(isinf(temp)) = log(1e-300);
        temp(isinf(temp)) = 0;
        
        %Evaluate this likelihood
        if L == 1
            lik = sum(pi_n.*temp);
        else
            lik = sum(pi_n.*sum(beta.*temp,2));
        end
       
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
      
        %Calculate BIC values
        if ~cmplx
            BIC(t,i) = lik - mo*log(sum(pi_n));
        else 
            %Removes + .0000000i term
            BIC(t,i) = real(lik) - mo*log(sum(pi_n));
        end
        
        %Check if BIC penality decreased, if so break from loop.
        if i > 1 && (BIC(t,i) - BIC(t,i-1)) < 0
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
    
    %output necessary posterior quantities.  
    if L == 1
        param.active_weights = 1;
    end
    
    ind = isnan(Rhat); Rhat(ind) = xhat(ind);
    
    gamma = repmat(EMfin.gamma_n,[1,optEM.Lmax,1]); gamma = permute(gamma,[1,3,2]);

    if L ~= 1
        beta = permute(EMfin.beta_n,[1,3,2])./repmat(sum(EMfin.beta_n,3),[1 L]);
    end
    
    beta(isnan(beta)) = 1/L;

    pi_n = EMfin.pi_n;

    
end

EMfin.BIC = BIC;

return
