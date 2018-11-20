function [param,uhatIter,plikeIt] = neuralGampEst(StimFull,outEst,...
    paramInit,Irow,gampOpt)
% Optimal estimation for GAMP

% Get options
rho = gampOpt.rho;
nit = gampOpt.nit;

% Get dimensions
[ndly,nin] = size(paramInit.linWt);

% Input estimation parameters
xinit = paramInit.p(1)*paramInit.linWt;
pinit = paramInit.p;
probAct = 0.5;
z0mean = probAct*sum(sum(xinit)) + pinit(2);
xinitMean = mean(mean(xinit));
xvar0 = mean(mean((xinit-xinitMean).^2))/rho;
z0var = 1;
nx = nin*ndly;
inEst = NeuralInEst(z0mean, z0var, repmat(xvar0,nx,1), ...
    rho, ndly, nin);

% Linear transform
relTol = 1e-3;
absTol = 1e-6;
StimFull.setTol( relTol, absTol );
linTrans = LinTransNeural(StimFull,Irow,ndly);

% GAMP parameters
opt = GampOpt();
opt.nit = nit;                % number of iterations
opt.step = 0.1;             % step size
opt.adaptStep = true;       % adaptive step size
opt.verbose = true;         % Print results in each iteration
opt.removeMean = true;      % Remove mean
opt.pvarMin = 0.01;
opt.xvarMin = 1e-5;

% Run GAMP
%[uhatIter, plikeIt] = neuralGampIter(inEst, outEst, linTrans, opt);
[uhatIter, plikeIt] = gampEst(inEst, outEst, linTrans, opt);

% Extract estimate
uhat = uhatIter(:,end);
xhat = reshape(uhat(1:nx), ndly, nin);
z0hat = uhat(nx+1);

% Find a polynomial fit for z
if 1
disp('Computing a polynomial fit');
nlFitOpt.np = 2;
nlFitOpt.niter = 50;
nlFitOpt.p0 = z0hat;

v = StimFull.firFilt(xhat, Irow);
p = nlFit(v,outEst,nlFitOpt);
end

param = NeuralParam(xhat,p,outEst.noiseVar);

