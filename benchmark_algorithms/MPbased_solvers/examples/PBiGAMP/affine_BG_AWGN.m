% This is an example of using P-BiG-AMP to estimate the sparse vectors 
% b and c from AWGN corrupted versions of the (affine) bilinear measurements
%   z(m) = b.'*squeeze(A(m,:,:))*c + Ac(m,:)*b + Ab(m,:)*c for m=1:M.
%
% In this example, b and c are drawn Bernoulli-Gaussian, the tensor A is
% drawn iid Gaussian, and the matrices Ab & Ac are also drawn iid Gaussian
% (or set to zero under "affine=false").  

% Handle random seed
if 1 % new random trial
    savedState = rng;
    save random_state.mat savedState;
else % repeat last trial
    load random_state.mat %#ok<UNRCH>
end
rng(savedState);

% Turn on/off plotting and verbosity
plotHist = 1; % figure to plot history (or zero for no plot)
verbose = true;

% Specify core problem parameters
affine = false; % use affine offset?
cmplx = false; % complex valued quantities?
SNRdB = 100; % SNR in dB

% Specify core problem dimensions
alpha = 0.75; % ratio of measurements-to-unknowns
sparsity = 0.25; % sparsity rate
N = 50; % length of unknown vectors 

% Compute ancillary problem dimensions
sparsity_b = sparsity; % sparsity rate
sparsity_c = sparsity; % sparsity rate
Nb = N; % length of c
Nc = N; % length of c
Kb = round(sparsity_b*Nb); % number of non-zeros in b 
Kc = round(sparsity_c*Nc); % number of non-zeros in c
M = round(alpha*(Nb+Nc)); % number of measurements

% Draw iid Bernoulli-Gaussian signal vectors
b = zeros(Nb,1);
c = zeros(Nb,1);
if cmplx
    b(randperm(Nb,Kb)) = sqrt(1/2)*complex(randn(Kb,1),randn(Kb,1));
    c(randperm(Nb,Kb)) = sqrt(1/2)*complex(randn(Kc,1),randn(Kc,1));
else
    b(randperm(Nb,Kb)) = randn(Kb,1);
    c(randperm(Nb,Kb)) = randn(Kc,1);
end

% Draw iid Gaussian measurement coefficients
if cmplx
    A = complex(randn(M,Nb,Nc),randn(M,Nb,Nc));
else
    A = randn(M,Nb,Nc);
end
if affine
    if cmplx
        Ac = sqrt(1/2)*complex(randn(M,Nb),randn(M,Nb));
        Ab = sqrt(1/2)*complex(randn(M,Nc),randn(M,Nc));
    else
        Ac = randn(M,Nb);
        Ab = randn(M,Nc);
    end
else
    Ac = [];
    Ab = [];
end

% Compute the noiseless measurements
z = ttv(ttv(tensor(A),c,3),b,2);
if ~isempty(Ac)
    %z = z + ttv(tensor(Ac),b,2);
    z = z + Ac*b;
end
if ~isempty(Ab)
    %z = z + ttv(tensor(Ab),c,2);
    z = z + Ab*c;
end
z = double(z);


% Create the noisy measurements
wvar = norm(reshape(z,[],1))^2/M*10^(-SNRdB/10);
wvar = max(wvar,1e-10);
if cmplx
    y = z + sqrt(wvar/2)*complex(randn(size(z)),randn(size(z)));
else
    y = z + sqrt(wvar)*randn(size(z));
end

% Declare the error functions
errfxnZ = @(qval) (norm(qval - z,'fro') / norm(z,'fro'))^2;
if affine
    errfxnB = @(qval) (norm(qval - b)/norm(b))^2;
    errfxnC = @(qval) (norm(qval - c)/norm(c))^2;
else
    % circumvent scalar ambiguity
    errfxnB = @(qval) (norm(qval*((qval'*b)/norm(qval)^2) - b)/norm(b))^2;
    errfxnC = @(qval) (norm(qval*((qval'*c)/norm(qval)^2) - c)/norm(c))^2;
end

% Set prior on B
if cmplx
    estInB = CAwgnEstimIn(0,1);
else
    estInB = AwgnEstimIn(0,1);
end
estInB = SparseScaEstim(estInB,sparsity_b);

% Set prior on C
if cmplx
   estInC = CAwgnEstimIn(0,1);
else
   estInC = AwgnEstimIn(0,1);
end
estInC = SparseScaEstim(estInC,sparsity_c);

% Set log likelihood
noisevar = mean(abs(y).^2)/(1+10^(SNRdB/10));
if cmplx
    estOut = CAwgnEstimOut(y, noisevar);
else
    estOut = AwgnEstimOut(y, noisevar);
end

% Initialize the options
opt = PBiGAMPOpt('nit',300,'step',0.3,'stepIncr',1.006,'stepMax',0.5,'adaptStep',false,'pvarStep',false,'zvarToPvarMax',inf,'tol',1e-7,'normTol',1e-10,'uniformVariance',true,'error_function',errfxnZ,'error_functionB',errfxnB,'error_functionC',errfxnC,'verbose',verbose);

% Initialize the means randomly
if cmplx
    opt.bhat0 = sqrt(1/2)*(randn(Nb,1) + 1j*randn(Nb,1));
    opt.chat0 = sqrt(1/2)*(randn(Nc,1) + 1j*randn(Nc,1));
else
    opt.bhat0 = randn(Nb,1);
    opt.chat0 = randn(Nc,1);
end

% Initialize the variances
[mB,vB,~] = estInB.estimInit(); opt.bvar0 = (vB+abs(mB).^2);
[mC,vC,~] = estInC.estimInit(); opt.cvar0 = (vC+abs(mB).^2);

% Call PBiGAMPsimple
if plotHist
    [estFin,optFin,estHist] = PBiGAMPsimple(estInB,estInC,estOut,A,Ac,Ab,opt);
else % faster if history not saved
    [estFin,optFin] = PBiGAMPsimple(estInB,estInC,estOut,A,Ac,Ab,opt);
end

% Compute final errors (in dB of NMSE)
errZdB = 10*log10(errfxnZ(estFin.zhat))
errBdB = 10*log10(errfxnB(estFin.bhat))
errCdB = 10*log10(errfxnC(estFin.chat))

% Plot results
if plotHist
    figure(plotHist); clf
    plot(10*log10(estHist.errZ),'b-x')
    hold on;
      plot(10*log10(estHist.errB),'r-x')
      plot(10*log10(estHist.errC),'g-x')
      plot(-SNRdB*ones(size(estHist.errZ)),'k--')
    hold off;
    legend('z error','b error','c error','success')
    grid
    xlabel('iteration')
    ylabel('NMSE [dB]')
end

if errZdB > -SNRdB
   display('Not successful!  Try running again...')
else
   display('successful!')
end
