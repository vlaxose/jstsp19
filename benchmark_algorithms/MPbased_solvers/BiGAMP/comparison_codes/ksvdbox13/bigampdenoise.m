function [y,D,nz] = bigampdenoise(params,msgdelta)
%BiG-AMP denoising
%  [Y,D] = BIGAMPDENOISE(PARAMS) denoises the specified (possibly
%  multi-dimensional) signal using BiG-AMP denoising. Y is the denoised
%  signal and D is the trained dictionary produced by BiG-AMP.
%
%
%
%   Summary of all fields in PARAMS:
%   --------------------------------
%
%   Required:
%     'x'                      signal to denoise
%     'blocksize'              size of block to process
%     'dictsize'               size of dictionary to train
%     'psnr' / 'sigma'         noise power in dB / standard deviation
%     'trainnum'               number of training signals
%
%   Optional (default values in parentheses):
%     'initdict'               initial dictionary ('odct')
%     'stepsize'               distance between neighboring blocks (1)
%     'iternum'                number of training iterations (10)
%     'maxval'                 maximal intensity value (1)
%     'noisemode'              'psnr' or 'sigma' ('sigma')
%     'maxatoms'               max # of atoms per block (prod(blocksize)/2)
%     'gain'                   noise gain (1.15)
%
%  Based on K-SVD Denoising code by: 
%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs



%%%%% parse input parameters %%%%%

x = params.x;
blocksize = params.blocksize;
trainnum = params.trainnum;
dictsize = params.dictsize;

p = ndims(x);
if (p==2 && any(size(x)==1) && length(blocksize)==1)
  p = 1;
end

% gain %
if (isfield(params,'gain'))
  gain = params.gain;
else
  gain = 1.15;
  params.gain = gain;
end

% blocksize %
if (numel(blocksize)==1)
  blocksize = ones(1,p)*blocksize;
end


% maxval %
if (isfield(params,'maxval'))
  maxval = params.maxval;
else
  maxval = 1;
  params.maxval = maxval;
end

% msgdelta %
if (nargin<2)
  msgdelta = 5;
end

verbose = 't';
if (msgdelta <= 0)
  verbose='';
  msgdelta = -1;
end


% initial dictionary %

if (~isfield(params,'initdict'))
  params.initdict = 'odct';
end

if (isfield(params,'initdict') && ischar(params.initdict))
  if (strcmpi(params.initdict,'odct'))
    params.initdict = odctndict(blocksize,dictsize,p);
  elseif (strcmpi(params.initdict,'data'))
    params = rmfield(params,'initdict');    % causes initialization using random examples
  else
    error('Invalid initial dictionary specified.');
  end
end

if (isfield(params,'initdict'))
  params.initdict = params.initdict(:,1:dictsize);
end


% noise mode %
if (isfield(params,'noisemode'))
  switch lower(params.noisemode)
    case 'psnr'
      sigma = maxval / 10^(params.psnr/20);
    case 'sigma'
      sigma = params.sigma;
    otherwise
      error('Invalid noise mode specified');
  end
elseif (isfield(params,'sigma'))
  sigma = params.sigma;
elseif (isfield(params,'psnr'))
  sigma = maxval / 10^(params.psnr/20);
else
  error('Noise strength not specified');
end

params.Edata = sqrt(prod(blocksize)) * sigma * gain;   % target error for omp
params.codemode = 'error';

params.sigma = sigma;
params.noisemode = 'sigma';


% make sure test data is not present in params
if (isfield(params,'testdata'))
  params = rmfield(params,'testdata');
end


%%%% create training data %%%

ids = cell(p,1);
if (p==1)
  ids{1} = reggrid(length(x)-blocksize+1, trainnum, 'eqdist');
else
  [ids{:}] = reggrid(size(x)-blocksize+1, trainnum, 'eqdist');
end
params.data = sampgrid(x,blocksize,ids{:});

% remove dc in blocks to conserve memory %
blocksize = 2000;
for i = 1:blocksize:size(params.data,2)
  blockids = i : min(i+blocksize-1,size(params.data,2));
  params.data(:,blockids) = remove_dc(params.data(:,blockids),'columns');
end



%%%%% BiG-AMP training %%%%%

if (msgdelta>0)
  disp('BiG-AMP training...');
end

%use BiG-AMP
%Specify sizes
[M,N] = size(params.initdict);
L = params.trainnum;
p1 = 1;
set_BiGAMP_options_script
EMopt.tmax = 20;
EMopt.maxEMiter = 5;
EMopt.maxEMiterInner = EMopt.maxEMiter;
opt.stepMax = 0.1;
opt.stepMin = 0.1;
opt.adaptStep = 0;
opt.nit = 100;
EMopt.init = params.initdict;
EMopt.learn_var = true;
EMopt.learn_mean = false;
EMopt.learn_lambda = true;
opt.verbose = 1;
EMopt.maxTol = 1e-2;
EMopt.warm_start = true;
EMopt.lambda = 0.5*ones(N,L);

%Error
opt.error_function = @(q)...
    20*log10(norm(q - params.data,'fro') / norm(params.data,'fro'));

%%%%%Run BGAMP
tstart = tic;
[~, ~, D] = ...
    EMBiGAMP_DL(params.data,opt,EMopt);
tEMGAMP = toc(tstart);

%D = params.initdict;

%plotUtilityNew(estHistEM,[-80 0],100,101)

%Normalize the columns of D
D = D * diag(1 ./ sqrt(diag(D'*D)));

%%%%%  denoise the signal  %%%%%

if (~isfield(params,'lambda'))
  params.lambda = maxval/(10*sigma);
end

params.dict = D;

if (msgdelta>0)
  disp('OMP denoising...');
end

% call the appropriate ompdenoise function
if (p==1)
  [y,nz] = ompdenoise1(params,msgdelta);
elseif (p==2)
  [y,nz] = ompdenoise2(params,msgdelta);
elseif (p==3)
  [y,nz] = ompdenoise3(params,msgdelta);
else
  [y,nz] = ompdenoise(params,msgdelta);
end

end
