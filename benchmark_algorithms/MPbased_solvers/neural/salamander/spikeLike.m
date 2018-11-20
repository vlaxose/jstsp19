function plike = spikeLike(StimFull, outEst, param, Irow)
% spikeLike:  Computes the likelihood of a spike sequence

% Compute the spike rate
v = StimFull.firFilt(param.linWt, Irow);
z = polyval( param.p, v);
nz = length(z);

plike = exp(outEst.logLike(z)/nz);

