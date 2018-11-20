% generatePRLinTrans: generates a GAMPmatlab LinTrans operator for use in phase retrieval
%
% A = generateLinTrans(nz,nx,A_type)
%
% where 
%   nz: output size  
%   nx: input size
%
% where A_type is one of
%     'iid': iid complex-Gaussian
%    'rdft': iid complex-Gaussian after DFT
%    'mdft': several nx-DFTs after ceil(nz/nx) binary masks
%    'odft': first nx columns of nz-by-nz DFT matrix 
%   'o2dft': first nx columns of nz-by-nz 2DFT matrix 
%   'm2dft': several 2DFTs after binary masks

function A = generatePRLinTrans(nz,nx,A_type,x)

  if strcmp(A_type,'iid'),                              % iid gaussian :)
    A = (1/sqrt(2*nx))*(randn(nz,nx)+1i*randn(nz,nx));
    A = MatrixLinTrans(A);
  elseif strcmp(A_type,'rdft'),                         % iidN after DFT :)
    F = dftmtx(nx)/sqrt(nx);
    A = (1/sqrt(2*nx))*(randn(nz,nx)+1i*randn(nz,nx))*F;
    A = MatrixLinTrans(A);
  elseif strcmp(A_type,'mdft'),                         % DFT after mask :)
    if nz<=nx, error('nz too small for masking!'); end;
    F = dftmtx(nx)/sqrt(nx);
    A = F;                      % first mask is trivial
    for i=1:ceil(nz/nx)-1,
      h = round(rand(nx,1));    % other masks are random binary 
      A = [A;F*diag(h)];
    end;
    A = A(1:nz,:);              % trim size
    A = MatrixLinTrans(A);
  elseif strcmp(A_type,'odft'),                         % oversampled DFT :(
    % note: shifts in x yield same abs(z)
    if nx>nz, error('nx too large for oversampling!'); end;
    %F = dftmtx(nz)/sqrt(nx);
    %A = MatrixLinTrans(F(:,1:nx));
    %
    %indx_y = 1:nz;
    %indx_x = 1:nx;
    %nfft = nz;
    %hA = @(x) PartialFFTlocal(x,1,nfft,indx_x,indx_y);
    %hAt = @(z) PartialFFTlocal(z,0,nfft,indx_x,indx_y);
    %A = FxnhandleLinTrans(length(indx_y),length(indx_x),hA,hAt);
    %
    hA = @(x) (1/sqrt(nz))*fft(x,nz);
    getfirstnx = @(x) x(1:nx);
    hAt = @(z) sqrt(nz)*getfirstnx(ifft(z));
    A = FxnhandleLinTrans(nz,nx,hA,hAt);
  elseif strcmp(A_type,'o2dft'),                        % oversampled 2DFT :|
    % note: shifts in x yield same abs(z)
    nX = sqrt(nx);
    ndft = sqrt(nz);
    if (ceil(nX)~=nX)||(ceil(ndft)~=ndft),
      error('nX or ndft is not an integer');
    end;
    A = sampTHzLinTrans(nX,nX,ndft,ndft);
  elseif strcmp(A_type,'m2dft'),                        % 2DFT after masks :)
    num_mask_min = 4;   % minimum # of masks [dflt>=3]
    nX = sqrt(nx);
    if ceil(nX)~=nX,
      error('nX is not an integer');
    end;
    ndft = sqrt(2^(2*ceil(log2(nx)/2)));        % ndft^2 >= nx, ndft=power-of-2
    num_masks = max(num_mask_min,ceil(nz/ndft^2));
    if 1,                                       % random binary
      Masks = round(rand(nX,nX,num_masks));
      Masks(:,:,num_masks) = 1-Masks(:,:,1);    % ensure an invertible transform
    else                                        % shuffled non-binary
      mask_gains = rand(1,num_masks);
      Masks = reshape( reshape(ones(nX,nX,1),nX^2,1)*(mask_gains(:).'),nX,nX,num_masks );
      for i=1:nX, for j=1:nX, Masks(i,j,:) = Masks(i,j,randperm(num_masks)); end; end;
    end;
    num_samp = ceil(nz/num_masks);
    if 1        % top left corner in Fourier space :)
      corner = zeros(ndft);
      corner(1:ceil(sqrt(num_samp)),1:ceil(sqrt(num_samp))) = ...
        ones(ceil(sqrt(num_samp)));
      indices = find(corner==1);
      SampLocs = indices(1:num_samp);
    else        % random indices :(
      SampLocs = nan(num_samp,num_masks);
      for k=1:num_masks,
        indices = randperm(ndft^2).';
        SampLocs(:,k) = sort(indices(1:num_samp));
      end;
    end;
    A = sampTHzLinTrans(nX,nX,ndft,ndft,SampLocs,Masks);
    nz = A.size;        % nz may have been rounded up
  else
    error('invalid A_type')
  end;

end % function

%-------------------------------------------------------------------
function out=PartialFFTlocal(in,mode,n,indx_x,indx_y) 

  if mode==1 % forward operator
    x = zeros(n,1);
    x(indx_x) = in;
    y = fft(x,n);
    out = y(indx_y(:));
  else % backward operator
    y = zeros(n,1);
    y(indx_y) = in;
    x = ifft(y,n)*n;
    out = x(indx_x(:));
  end

end % function
