function p = nlFit(v,outEst,opt)
% nlFit:  Fits a polynomial to count data
%
% Model is:
%   v = StimFull.firFilt(param.linWt)
%   z = polyval( param.poly, v);
%   rate = log(1 + exp(v))
%   cnt = poissrnd(rate)
%

% Get options
niter = opt.niter;  % number of iterations
np = opt.np;        % number of polynomial terms
p0 = opt.p0; % Optimal z for a constant input

% Do a coarse initial search
np1 = 20;
p1Test = linspace(0,2,np1)';
plikeTest = zeros(np1,1);
for ip1 = 1:np1
    p1 = p1Test(ip1);
    p = [p1; p0];
    plikeTest(ip1) = -logLike(p,v,outEst);
    fprintf(1,'p1=%f plike=%f\n', p1, plikeTest(ip1));
end
[mm,im] = max(plikeTest);
pinit = [p1Test(im) p0];

% Run optimizer
fminopt = optimset('Display','iter', 'MaxIter', niter, 'TolFun', 0.001, 'TolX', 0.1);
p = fminsearch(@(p) logLike(p,v,outEst), pinit, fminopt);

end

% Negative log likelihood for minimization
function plike = logLike(p, v, outEst)
    
    % Polynomial 
    z = polyval(p,v);
    nz = length(z);
    
    % Likelihood
    plike = -exp(outEst.logLike(z)/nz);
end




