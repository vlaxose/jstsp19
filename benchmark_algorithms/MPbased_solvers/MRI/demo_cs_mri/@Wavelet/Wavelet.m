function res = Wavelet(filterType, filterSize, wavScale)
% res = Wavelet(Filtertype, filterSize, wavScale)
%
% implements a wavelet operator
%
% (c) Michael Lustig 2007

res.adjoint = 0;
res.qmf = MakeONFilter(filterType, filterSize);
res.wavScale = wavScale;
res = class(res,'Wavelet');
