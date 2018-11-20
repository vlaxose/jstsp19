% MRITest:  Test program for the MRI GAMP reconstruction
% -------------------------------------------------------
% Compares GAMP reconstruction against conjugate gradient (CG) optimization

% Set paths
addpath('../main/');
addpath('demo_cs_mri');
addpath(genpath('c:\program Files\MATLAB\R2011a\toolbox\Wavelab850'));

% Run the standard conjugate gradient (CG) CS method if requested,
% or if the program has not been run once.  The CG method, which is
% supplied in the Stanford code provieds the baseline for comparison.
runCG = false;
stdReconExist = (exist('recon_full', 'var') && ...
    exist('recon_full', 'var') && exist('recon_full', 'var'));
if (runCG || ~stdReconExist)
    disp('Running standard CS method');
    example1_2d_brain_mri_cg;
    save example1_results recon_full recon_dft recon_cs param;
else
    disp('Loading standard CS results');
    load example1_results;
end

% Get dimensions
nrow = size(param.y,1);
ncol = size(param.y,2);
ncoil = size(param.y,3);
nx = nrow * ncol;        % Number of input components

% Input estimator.  In this case, use a null estimator
xmean0 = 0;
xvar0 = 4;
mriEstIn = NullEstimIn(repmat(xmean0,nx,1), repmat(xvar0,nx,1) );

% Output estimator describing p(y|z)
autoScale = false;
mriEstOut = MriEstimOut(param, autoScale);

% Create linear transform
mriLT = MriLinTrans(param);

% GAMP parameters.  See class GampOpt
numiter = 20;           % Number of iterations
opt = GampOpt();        % default parameters
opt.step = 1;           % step size
opt.nit = numiter;      % number of iterations
opt.removeMean = true;  % remove mean
opt.pvarMin = 0.0001;
opt.xvarMin = 0.0001;
opt.adaptStep = true;
opt.verbose = true;
opt.tol = -1;           % do not allow early termination
opt.stepTol = -1;       % do not allow early termination
opt.stepWindow = 1;

% Call the GAMP algorithm
% The function returns xhatTot(i,t) = estimate of x(i) on iteration t.
% So, xhatTot(:,end) is the final estimate
[xhat, xvar, rhat, rvar, shatFinal, svarFinal,zhatFinal,zvarFinal, estHist] = ...
    gampEst(mriEstIn, mriEstOut, mriLT, opt);
xhatGAMP = xhat;

% Plot image
xsq = reshape(xhat,nrow,ncol);
subplot(1,3,1);
imshow(abs(recon_full));
title('Fully sampled');
subplot(1,3,2);
imshow(abs(recon_cs));
title('CG');
subplot(1,3,3);
imshow(abs(xsq));
title('GAMP');

% Correlation
rhoGAMP = abs(sum(sum((conj(xsq).*recon_full))))^2/...
    norm(xsq,'fro')^2/norm(recon_full,'fro')^2;
rhoCG = abs(sum(sum((conj(recon_cs).*recon_full))))^2/...
    norm(recon_cs,'fro')^2/norm(recon_full,'fro')^2;
fprintf(1,'Correlation CG:   %f\n', rhoCG);
fprintf(1,'Correlation GAMP: %f\n', rhoGAMP);

