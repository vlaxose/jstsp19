% polyFitCdf:  Fits a polynomial to obtain an inverse CDF.
%
% Suppose that z ~ N(zmean, zvar), and v is the polynomial function:
%
%   v = exp(polyval(lambda,u)), u = 1/(1+exp(-z))
%
% Then, polyFitCdf finds parameters lambda such that the inverse CDF of V
% approximately matches a target inverse CDF at:
%   
%   F_V^{-1}(t) = Finv(t)  for all t.
function [lam,z,Finvapp] = polyFitCdf(zmean, zvar, npoly, tinv, Finv)

% Compute z=F_Z^{-1}(tinv) and corresponding values for u
w = sqrt(2)*erfinv(2*tinv-1);
z = zmean + sqrt(zvar)*w;
u = 1./(1+exp(-z));

% Fit polynomial  
nz = length(tinv);
U = ones(nz,npoly);
for ifit = npoly-1:-1:1
    U(:,ifit) = U(:,ifit+1).*u;
end
lam = U \ log( max(1, Finv) );

% Compute approx CDF
Finvapp = exp( U*lam );

return
