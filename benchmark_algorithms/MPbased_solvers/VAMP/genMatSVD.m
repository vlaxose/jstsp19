% out = genMatSVD(M,N,UType,svType,svParam,VType,['Property',value])
%
% Generate an MxN matrix with the 'economy' SVD form 
%    A = U*diag(s)*V' 
% where
%    s is non-zero and length min(M,N) ... "singular values"
%    U is size M x min(M,N) and obeys U'*U = I ... "left singular vectors"
%    V is size N x min(M,N) and obeys V'*V = I ... "right singular vectors"
%
% inputs:
%   M : # rows in A
%   N : # columns in A
%   UType : type of U in {'Haar','DFT','DCT','DHT','DHTrice','I'}
%           'Haar' means uniformly drawn from all orthogonal/unitary matrices
%   svType : type of s in {'cond_num','spread','low_rank'}
%   svParam : parameter controlling s
%      svType=cond_num => svParam=cond(A) 
%      svType=spread => s= (iid Gaussian singular values).^svParam 
%      svType=low_rank => svParam=rank(A) 
%   VType : type of V in {'Haar','DFT','DCT','DHT','DHTrice','I'}
%
% optional properties
%   'Afro2' : specify squared Frobenius norm of A.  [default=N]
%   'fxnHandles' : generate fxn handles for A,U,V?  [default=false]
%   'isCmplx' : generate complex-valued matrices?  [default=false (unless DFT)]
%   'shuffle' : randomly permute rows of V'?  [default=true]
%   'randsign' : randomly flip signs on columns of V'?  [default=true]
%
% output if fxnHandles=false:
%   out.U = U;
%   out.V = V;
%   out.s = s;
%   out.A = U*(spdiags(s,0,M,N)*V');
%
% output if fxnHandles=true:
%   out.fxnU = fxnU;
%   out.fxnUh = fxnUh;
%   out.fxnS = fxnS;
%   out.fxnSh = fxnSh;
%   out.fxnV = fxnV;
%   out.fxnVh = fxnVh;
%   out.s = s;
%   out.fxnA = @(x) fxnU(fxnS(fxnVh(x)));
%   out.fxnAh = @(z) fxnV(fxnSh(fxnUh(z)));

function out = genMatSVD(M,N,UType,svType,svParam,VType,varargin)

% check inputs
useDHTrice = false;
if strcmp(UType,'DHTrice')||strcmp(VType,'DHTrice')
  useDHTrice = true;
end
useDFT = false;
if strcmp(UType,'DFT')||strcmp(VType,'DFT')
  useDFT = true;
end
if useDHTrice&&useDFT
  error('Cannot combine DFT and DHTrice since latter does not allow complex numbers')
end

% set default options
opt.Afro2 = N;
opt.fxnHandles = false;
opt.isCmplx = false;
if useDFT
  opt.isCmplx = true;
end
opt.shuffle = true;
opt.randsign = true;

% overwrite options with user-provided values
if nargin>7
  for i = 1:2:length(varargin)
    opt.(varargin{i}) = varargin{i+1};
  end
end

% check options
if useDHTrice & opt.isCmplx
  error('DHTrice does not allow complex numbers')
end
if strcmp(VType,'Haar')
  opt.shuffle = false; % not allowed since we use 'econ' SVD!
  opt.randsign = false; % not necessary
end

% setup for Haar or spread cases 
if strcmp(UType,'Haar')||strcmp(VType,'Haar')||strcmp(svType,'spread')
  if opt.isCmplx
    G = randn(M,N)+1i*randn(M,N);
  else
    G = randn(M,N); 
  end
  if M<=N
    [Uhaar,D] = eig(G*G'); % G=U*S*V' with tall V
  else % M>N
    [Vhaar,D] = eig(G'*G); % G=U*S*V' with tall U
  end
  d = diag(D); % d = s.^2
end;

% generate singular values
switch svType
  case 'spread'
    spread = svParam;
    s = d.^(spread/2); % spread the singular values

  case 'cond_num'
    cond_num = svParam;
    R = min(M,N);
    ratio = cond_num^(1/(R-1)); % ratio between consecutive values
    s = ratio.^[R:-1:1]; % geometric series

  case 'low_rank'
    R = svParam; % low rank
    s = [ones(1,R),eps*ones(1,min(M,N)-R)]; % use eps for numerical reasons

  otherwise
    error('unrecognized svType')

end
s = s(:)./sqrt(sum(s.^2)/opt.Afro2); % normalize so that norm(A,'fro')^2=Afro2

% generate left singular value matrix
switch UType 
  case 'I'
    if opt.fxnHandles
      fxnU = @(z) z;
      fxnUh = @(z) z;
    else
      U = speye(M);
    end

  case 'Haar'
    if M<=N
      U = Uhaar; % square
    else % M>N
      Utall = G*(Vhaar*spdiags(1./sqrt(d),0,N,N)); % tall MxN
    end
    if opt.fxnHandles
      if M<=N
        fxnU = @(z) U*z;
        fxnUh = @(z) U'*z;
      else % leverage M>N
        fxnU = @(z) Utall*z(1:N,:); % tall U
        fxnUh = @(z) speye(M,N)*(Utall'*z); % tall U
      end
    else
      if M>N
        U = [Utall,zeros(M,M-N)]; % not unitary, but should never use zeros
      end
    end

  case 'DFT'
    if opt.fxnHandles
      fxnU = @(z) fft(z,M)*(1/sqrt(M));
      fxnUh = @(z) ifft(z,M)*sqrt(M);
    else
      U = dftmtx(M)*(1/sqrt(M)); % unitary MxM
    end

  case 'DCT'
    if opt.fxnHandles
      fxnU = @(x) dct(x,M);
      fxnUh = @(z) idct(z,M);
    else
      U = dct(eye(M)); % unitary MxM
    end

  case 'DHT' % slow but handles matrix input
    if (floor(log2(M))~=log2(M))
      error('For DHT, M must be a multiple of two.')
    end
    if opt.fxnHandles
      fxnU = @(z) fwht(z,M,'hadamard')/sqrt(M); 
      fxnUh = @(z) ifwht(z,M,'hadamard')*(1/sqrt(M)); 
    else
      U = hadamard(M)/sqrt(M); % unitary MxM
    end

  case 'DHTrice' % fast but only handles vector input
    if (floor(log2(M))~=log2(M))
      error('For DHTrice, M must be a multiple of two.')
    end
    if opt.fxnHandles
      fxnU = @(z) fWHtrans(z)*sqrt(M); 
      fxnUh = @(z) ifWHtrans(z)*(1/sqrt(M)); 
    else
      U = zeros(M); IM = eye(M);
      for m=1:M, U(:,m) = fWHtrans(IM(:,m))*sqrt(M); end;
    end

  otherwise
    error('unrecognized UType')
end

% generate right singular value matrix
switch VType 
  case 'I'
    if opt.fxnHandles
      fxnV = @(x) x;
      fxnVh = @(x) x;
    else
      V = speye(N);
    end

  case 'Haar'
    if M<=N
      Vtall = G'*(Uhaar*spdiags(1./sqrt(d),0,M,M)); % tall NxM
    else % M>N
      V = Vhaar; % square
    end
    if opt.fxnHandles
      if M<=N
        fxnV = @(x) Vtall*x(1:M,:);
        fxnVh = @(x) speye(N,M)*(Vtall'*x);
      else % M>N
        fxnV = @(x) V*x;
        fxnVh = @(x) V'*x;
      end
    else
      if M<=N
        V = [Vtall,zeros(N,N-M)]; % not unitary, but should never use zeros
      end
    end

  case 'DFT'
    if opt.fxnHandles
      fxnV = @(x) fft(x,N)*(1/sqrt(N));
      fxnVh = @(x) ifft(x,N)*sqrt(N);
    else
      V = dftmtx(N)*(1/sqrt(N)); % unitary NxN
    end

  case 'DCT'
    if opt.fxnHandles
      fxnV = @(x) dct(x,N);
      fxnVh = @(z) idct(z,N);
    else
      V = dct(eye(N)); % unitary NxN
    end

  case 'DHT'
    if (floor(log2(N))~=log2(N))
      error('For DHT, N must be a multiple of two.')
    end
    if opt.fxnHandles
      fxnV = @(x) fwht(x,N,'hadamard')*sqrt(N); 
      fxnVh = @(x) ifwht(x,N,'hadamard')*(1/sqrt(N)); 
    else
      V = hadamard(N)/sqrt(N); % unitary NxN
    end

  case 'DHTrice'
    if (floor(log2(N))~=log2(N))
      error('For DHTrice, N must be a multiple of two.')
    end
    if opt.fxnHandles
      fxnV = @(x) fWHtrans(x)*sqrt(N); 
      fxnVh = @(x) ifWHtrans(x)*(1/sqrt(N)); 
    else
      V = zeros(N); IN = eye(N);
      for n=1:N, V(:,n) = fWHtrans(IN(:,n))*sqrt(N); end;
    end

  otherwise
    error('unrecognized VType')
end

% shuffle rows of V' (i.e., columns of V)?
if opt.shuffle
  randpermN = randperm(N);
  if opt.fxnHandles
    IN = speye(N); J = IN(:,randpermN);
    fxnV = @(x) fxnV(J*x);
    fxnVh = @(x) J'*fxnVh(x);
  else
    V = V(:,randpermN);
  end
end

% randomly flip signs of columns of V' (i.e., rows of V)?
if opt.randsign
  if opt.isCmplx
    signvec = sign(randn(N,2)*[1;1i]);
  else
    signvec = sign(randn(N,1));
  end
  if opt.fxnHandles
    fxnV = @(x) bsxfun(@times,signvec,fxnV(x));
    fxnVh = @(x) fxnVh(bsxfun(@times,conj(signvec),x));
  else
    V = bsxfun(@times,signvec,V);
  end
end

% fill output structure
if opt.fxnHandles
  if M<=N
    fxnS = @(x) bsxfun(@times,s,x(1:M,:)); % wide
    %fxnSh = @(z) [bsxfun(@times,s,z);zeros(N-M,size(z,2))]; % tall
    fxnSh = @(z) spdiags(s,0,N,M)*z; % tall
  else
    fxnSh = @(z) bsxfun(@times,s,z(1:N,:)); % wide
    %fxnS = @(x) [bsxfun(@times,s,x);zeros(M-N,size(x,2))]; % tall
    fxnS = @(x) spdiags(s,0,M,N)*x; % tall
  end
  out.fxnA = @(x) fxnU(fxnS(fxnVh(x)));
  out.fxnAh = @(z) fxnV(fxnSh(fxnUh(z)));
  out.fxnU = fxnU;
  out.fxnUh = fxnUh;
  out.fxnS = fxnS;
  out.fxnSh = fxnSh;
  out.fxnV = fxnV;
  out.fxnVh = fxnVh;
  out.s = s;
else
  if M<=N,
    out.A = (U*spdiags(s,0,M,M))*V(:,1:M)'; % square x square x wide
  else
    out.A = U(:,1:N)*(spdiags(s,0,N,N)*V'); % tall x square x square
  end
  out.U = U;
  out.V = V;
  out.s = s;
end


