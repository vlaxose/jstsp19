classdef bandTHzLinTrans < LinTrans
    % bandTHzLinTrans:  Linear transform class for the THz application. 
    %   Here, a 2D signal (of size Nsig1 x Nsig2) is first pointwise 
    %   multiplied by a sequence of Mmask "mask" matrices (each of size 
    %   Nsig1 x Nsig2) and then each output is tranformed by a unitary 
    %   2D-DFT of size Ndft1 x Ndft2, which involves zero-padding if the 
    %   DFT dimensions are larger than the signal dimensions.  Finally,
    %   each (vectorized) FFT output is multiplied by a dimensionality 
    %   reducing banded matrix of size Msamp x Ndft1*Ndft2, whose 
    %   diagonals are specified in a matrix of size Ndft1*Ndft2 x Mband.
    
    properties
	Nsig1;	% (Nsig1 x Nsig2)-size 2D image signal
	Nsig2;
	Ndft1;  % (Ndft1 x Ndft2)-size Fourier tranform
	Ndft2;
	Msamp;	% number of output samples per Mask
	Mband;	% number of bands
	Band;	% (Ndft1*Ndft2 x Mband)-array of output mixing coefficients
	Mmask;	% number of masks
	Mask;   % (Nsig1 x Nsig2 x Mmask)-array of image-domain mask gains
%	SqMask; % (Nsig1 x Nsig2 x Mmask)-array of image-domain mask |gains|^2
%	SampLocs;% (Msamp x Mmask)-array of Fourier-domain indices  
	Fro2;	% squared Frobenius norm of operator
    end
    
    methods
        
        % Constructor
        function obj = bandTHzLinTrans(Nsig1,Nsig2,Ndft1,Ndft2,Msamp,Band,Mask)
            obj = obj@LinTrans;

	    if (Ndft1 < Nsig1)||(Ndft2 < Nsig2), 
	      error('DFT dimensions must be at least as large as signal dimensions'); 
	    end 
	    obj.Nsig1 = Nsig1;	
	    obj.Nsig2 = Nsig2;	
	    obj.Ndft1 = Ndft1;	
	    obj.Ndft2 = Ndft2;	

            if nargin > 4
	      if Msamp > Ndft1*Ndft2
                error('Need Msamp <= Ndft1*Ndft2')
	      end;
              obj.Msamp = Msamp;
	    else
	      obj.Msamp = Ndft1*Ndft2;
	    end;

	    if nargin > 5
	      obj.Band = Band;	
	      if size(Band,1)~=Ndft1*Ndft2
                error('Number of rows in Band must equal Ndft1*Ndft2')
	      end;
	      obj.Mband = size(Band,2);
            else
              obj.Band = ones(Ndft1*Ndft2,1);
	      obj.Mband = 1;
            end 

	    if nargin > 6 
	      obj.Mask = Mask;	
	      %obj.SqMask = abs(Mask).^2; 
            else
  	      obj.Mask = ones(Nsig1,Nsig2);
  	      %obj.SqMask = ones(Nsig1,Nsig2);
            end;
	    obj.Mmask = size(obj.Mask,3);

	    % approximate the squared Frobenius norm
	    numProbe = 10;	% increase for a better approximation
            obj.Fro2 = 0;      % initialize
	    for i=1:numProbe,
	      x = randn(obj.Nsig1*obj.Nsig2,1); % unit-variance probing signal
	      obj.Fro2 = obj.Fro2 + sum(abs(obj.mult(x)).^2)/numProbe;
	    end;
        end
        
        % Size
        function [m,n] = size(obj)
	    n = obj.Nsig1*obj.Nsig2;	
	    m = obj.Msamp*obj.Mmask;
        end
        
        % Matrix multiply
        function y = mult(obj,x)
	    X = reshape(x,obj.Nsig1,obj.Nsig2);
	    y = zeros(obj.Msamp,obj.Mmask);
            num = ceil((obj.Ndft1*obj.Ndft2+obj.Mband-1)/obj.Msamp);
	    for k=1:obj.Mmask,
	      Yk = fft2( X.*obj.Mask(:,:,k)*(1/sqrt(obj.Ndft1*obj.Ndft2)), ...
			obj.Ndft1, obj.Ndft2 );
              BYk = zeros(num*obj.Msamp,1);
              for l=1:obj.Mband,
		BYk = BYk + [zeros(l-1,1);obj.Band(:,l).*Yk(:);zeros(num*obj.Msamp-obj.Ndft1*obj.Ndft2-l+1,1)];
              end;
              y(:,k) = reshape(BYk,obj.Msamp,num)*ones(num,1);
	    end;        
	    y = y(:);
        end

        % Hermitian-transposed-Matrix multiply 
        function x = multTr(obj,y)
	    x = zeros(obj.Nsig1,obj.Nsig2);
	    Y = reshape(y,obj.Msamp,obj.Mmask);
            num = ceil((obj.Ndft1*obj.Ndft2+obj.Mband-1)/obj.Msamp);
	    for k=1:obj.Mmask,
	      yyk = Y(:,k)*ones(1,num);
	      yyk = yyk(1:obj.Ndft1*obj.Ndft2+obj.Mband-1).';
	      yk = zeros(obj.Ndft1*obj.Ndft2,1);
	      for l=1:obj.Mband,
	        yk = yk + conj(obj.Band(:,l)).*yyk(l:obj.Ndft1*obj.Ndft2+l-1);
	      end;
	      Xk = (ifft2(reshape(yk,obj.Ndft1,obj.Ndft2))*sqrt(obj.Ndft1*obj.Ndft2));
	      x = x + Xk(1:obj.Nsig1,1:obj.Nsig2).*conj(obj.Mask(:,:,k));
	    end;
	    x = x(:);
        end
        
        
        % Squared-Matrix multiply 
        function y = multSq(obj,x)
	    [m,n] = obj.size();
	    y = ones(m,1)*((obj.Fro2/m)*mean(x,1));
        end
        
        
        % Squared-Hermitian-Transposed Matrix multiply 
        function x = multSqTr(obj,y)
	    [m,n] = obj.size();
	    x = ones(n,1)*((obj.Fro2/n)*mean(y,1));
        end
        
    end
end
