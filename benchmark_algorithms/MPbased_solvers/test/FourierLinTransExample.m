%Simple Test code for FourierLinTrans
clear all
clc

% Set path
addpath('../main/');

%Set dimension, 1 or 2 for plots, class supports any Q. Computation is
%VERY large for Q > 3.
Q = 2;
K = 15; %number of non-zeros
SNR = 20;
subsample_factor = .05; %Set to 1 for full DFT
domain = true; %set to true for DFT A matrix, false for IDFT

%Set sizes
ySize = 128*ones(1,Q);
xSize = 128*ones(1,Q);


%Build the object
A = FourierLinTrans(ySize,xSize,domain);

%Compute random sample locations
A.ySamplesRandom(floor(subsample_factor*prod(ySize)));

%Get sizes
[m,n] = A.size();

%verify that subscript functions work correctly
safety = A.ySamples;
subVals = A.ySamplesSubscripts;
A.ySamplesSetFromSubScripts(subVals);
check = sum(safety - A.ySamples);
disp(['Check on subscript setup should be zero: ' num2str(check)])


%Draw true signal
whichOn = randperm(n);
whichOn = whichOn(1:K);
x = zeros(n,1);
x(whichOn) = sqrt(1/2)*(randn(K,1) + 1j*randn(K,1));


%Compute y
y = A.mult(x);
muw = norm(y)^2/m*10^(-SNR/10);
y = y + sqrt(muw/2)*(randn(m,1) + 1j*randn(m,1)); %noise

%Compute the Hermetian
AHAx = A.multTr(y);

%Use GAMP
opt = GampOpt; %initialize the options object

%Input channel
xmean0 = 0;
xvar0 = 1;
inputEst = CAwgnEstimIn(xmean0, xvar0);
inputEst = SparseScaEstim(inputEst,K/n);

%Output channel
outputEst = CAwgnEstimOut(y, muw);

%Use GAMP
xhat = gampEst(inputEst, outputEst, A, opt);


%Show result
if Q == 2 %2D image
    figure(1)
    clf
    imagesc(reshape(abs(x),xSize),[0 1]);
    colorbar
    title('x')
    
    figure(2)
    clf
    imagesc(reshape(abs(AHAx),xSize),[0 1])
    colorbar
    
    title('A^HAx')
    
    figure(3)
    clf
    imagesc(reshape(abs(xhat),xSize),[0 1])
    colorbar
    title(['GAMP Result, NMSE= '...
        num2str(20*log10(norm(xhat-x)/norm(x))) ' dB'])
    
else %for 1D or 3D or larger, just plot the coefficients
    figure(1)
    clf
    stem(abs(x))
    title('x')
    
    figure(2)
    clf
    stem(abs(AHAx))
    title('A^HAx')
    
    figure(3)
    clf
    stem(abs(xhat))
    title(['GAMP Result, NMSE= '...
        num2str(20*log10(norm(xhat-x)/norm(x))) ' dB'])
    
    
end






