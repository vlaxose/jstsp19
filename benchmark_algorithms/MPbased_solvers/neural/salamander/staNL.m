function param = staNL(StimFull,outEst,ndly,nlFitOpt,Irow)
% staNL:  Returns parameters in LNP model for STA
%
% Model is:
%   z = StimFull.firFilt(param.linWt) + N(0,noiseVar)
%   v = polyval( param.poly, z);
%   rate = log(1 + exp(v))
%   cnt = poissrnd(rate)
%
% The model orders are 
%   ndly = number of delays used in linear weigths
%   np = number of polynomial terms (right now np=2)


% Compute the STA subspace, normalized 
disp('Computing the STA...');
cnt1 = outEst.cnt;
linWt = StimFull.firFiltTr(cnt1, ndly, Irow);
linWt = linWt - mean(mean(linWt));
scale = sum(sum(linWt.^2));
linWt = linWt / sqrt(scale);

% Find a polynomial fit for zsta 
disp('Computing a polynomial fit');
v = StimFull.firFilt(linWt, Irow);
p = nlFit(v,outEst,nlFitOpt);

% Pack parameters
param = NeuralParam(linWt,p,outEst.noiseVar);
