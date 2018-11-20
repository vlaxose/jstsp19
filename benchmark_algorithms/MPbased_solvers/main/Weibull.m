function [x0,px0] = Weibull(k, lambda, xmax, nx0)

% Set defaults
if (nargin < 3)
    xmax = 10;
end
if (nargin < 4)
    nx0 = 1000;
end

% Generate pdf
x0 = linspace(0, xmax, nx0)';
px0 = wblpdf(x0+xmax/(2*nx0),lambda,k);
px0 = px0 / sum(px0);
