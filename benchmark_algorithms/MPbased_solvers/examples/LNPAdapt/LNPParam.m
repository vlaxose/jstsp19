% LNPParam:  Parameter file for the adaptive LNP problem.
%
% This file is called before both the simulation and SE analysis program.
%
% In the LNP problem, x is a Bernoulli-Gaussian random vector, that we want to
% estimate from measurements of the form
%
%   y = Poisson(v)  v = f(u), u=1/(1+exp(-z)) z=Ax
%
% and 
%
%   f(z)=exp(polyval(z,lambda) )
%
% for some polynomial coefficients lambda.
%
% Adaptive GAMP will estimate the parameters lambda along with estimating
% x.  The "oracle" GAMP, that serves as a point of comparison, uses the
% true lamdbda value.

% Sets path
addpath('../../main/');
addpath('../../stateEvo/');

% Parameters
nx = 1000;        % Number of input components (dimension of x)
nz = 500;         % Number of output components (dimension of y)
sparseRat = 0.1;  % fraction of components of x that are non-zero
xmean1 = 0;       % Input mean and variance (prior to sparsification)
xvar1 = 1;
nit = 20;         % number of iterations of GAMP
ntest = 50;       % number of Monte Carlo tests

% Rate function parameters.  The rate function is a polynomial fitting a
% saturation.
npoly = 3;      % Number of polynomial terms
vlo = 10;       % Low rate
vhi = 200;      % High rate
plo = 0.1;      % Prob(z <= zlo) where rate(zlo) = vlo
phi = 0.7;      % Prob(z >= zhi) where rate(zhi) = vhi
tinv = linspace(0.01,0.99,100)';
Finv = min(max( (tinv-plo)*(vhi-vlo)/(phi-plo) + vlo, ...
    vlo), vhi );

% Find polynomial approximation as the "true" parameters
xvar0 = sparseRat*xvar1;
zmean0 = 0;
zvar0 = xvar0;
[lamTrue,zapp,Finvapp] = polyFitCdf(zmean0, zvar0, npoly, tinv, Finv);

% Plot the rate CDF
if 1
    subplot(1,1,1);
    plot(tinv, [Finv Finvapp]);
    grid on;
end

% Generate input estimation class
inputEst0 = AwgnEstimIn(xmean1, xvar1);
inputEst = SparseScaEstim( inputEst0, sparseRat );
