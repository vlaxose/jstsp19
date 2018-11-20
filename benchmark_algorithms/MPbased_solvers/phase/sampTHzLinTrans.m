classdef sampTHzLinTrans < LinTrans
    % sampTHzLinTrans:  Linear transform class for the THz application. 
    %   Here, a 2D signal (of size Nsig1 x Nsig2) is first pointwise 
    %   multiplied by a sequence of Mmask "mask" matrices (each of size Nsig1 
    %   Nsig1 x Nsig2) and then each output is tranformed by a unitary 2D-DFT 
    %   of size Ndft1 x Ndft2, which involves zero-padding if the DFT 
    %   dimensions are larger than the signal dimensions.  Finally,
    %   each FFT output is subsampled at the locations in the corresponding
    %   column of SampLocs (of size Msamp x Mmask). 
    
    properties
	Nsig1;	% (Nsig1 x Nsig2)-size 2D image signal
	Nsig2;
	Ndft1;  % (Ndft1 x Ndft2)-size Fourier tranform
	Ndft2;
	Mmask;	% number of masks
	Mask;   % (Nsig1 x Nsig2 x Mmask)-array of image-domain mask gains
	SqMask; % (Nsig1 x Nsig2 x Mmask)-array of image-domain mask |gains|^2
	Msamp;	% length of SampLocs
	SampLocs;% (Msamp x Mmask)-array of Fourier-domain indices  
    end
    
    methods
        
        % Constructor
        function obj = sampTHzLinTrans(Nsig1,Nsig2,Ndft1,Ndft2,SampLocs,Mask)
            obj = obj@LinTrans;

	    if (Ndft1 < Nsig1)||(Ndft2 < Nsig2), 
	      error('DFT dimensions must be at least as large as signal dimensions'); 
	    end 
	    obj.Nsig1 = Nsig1;	
	    obj.Nsig2 = Nsig2;	
	    obj.Ndft1 = Ndft1;	
	    obj.Ndft2 = Ndft2;	

	    if nargin > 4
	      obj.SampLocs = SampLocs;	
            else
              obj.SampLocs = 1:(obj.Ndft1*obj.Ndft2);
            end 

	    if nargin > 5 
	      obj.Mask = Mask;	
	      obj.SqMask = abs(Mask).^2; 
            else
  	      obj.Mask = ones(Nsig1,Nsig2);
  	      obj.SqMask = ones(Nsig1,Nsig2);
            end;
	    obj.Mmask = size(obj.Mask,3);

	    if min(size(obj.SampLocs))==1,
	      obj.SampLocs = obj.SampLocs(:)*ones(1,obj.Mmask);
	    elseif size(obj.SampLocs,2) ~= obj.Mmask
              error('Number of columns in SampLocs must equal the number of masks')
	    end;
	    obj.Msamp = size(obj.SampLocs,1);
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
	    for k=1:obj.Mmask,
	      Yk = fft2( X.*obj.Mask(:,:,k)*(1/sqrt(obj.Ndft1*obj.Ndft2)), obj.Ndft1, obj.Ndft2 );
  	      y(:,k) = Yk(obj.SampLocs(:,k));
	    end;        
	    y = y(:);
        end

        % Hermitian-transposed-Matrix multiply 
        function x = multTr(obj,y)
	    x = zeros(obj.Nsig1,obj.Nsig2);
	    yk = reshape(y,obj.Msamp,obj.Mmask);
	    for k=1:obj.Mmask,
	      Yk = zeros(obj.Ndft1,obj.Ndft2);
	      Yk(obj.SampLocs(:,k)) = yk(:,k);
	      Xk = ifft2(Yk)*sqrt(obj.Ndft1*obj.Ndft2);
	      x = x + Xk(1:obj.Nsig1,1:obj.Nsig2).*conj(obj.Mask(:,:,k));
	    end;
	    x = x(:);
        end
        
        
        % Squared-Matrix multiply 
        function y = multSq(obj,x)
	    X = reshape(x,obj.Nsig1,obj.Nsig2);
	    y = zeros(obj.Msamp,obj.Mmask);
	    for k=1:obj.Mmask,
	      Yk = sum(sum(X.*obj.SqMask(:,:,k)))/(obj.Ndft1*obj.Ndft2);
              y(1:obj.Msamp,k) = Yk*ones(obj.Msamp,1);
            end;
            y = y(:);
        end
        
        
        % Squared-Hermitian-Transposed Matrix multiply 
        function x = multSqTr(obj,y)
            x = zeros(obj.Nsig1,obj.Nsig2);
	    yk = reshape(y,obj.Msamp,obj.Mmask);
	    for k=1:obj.Mmask,
	      x = x + obj.SqMask(:,:,k)*sum(yk(:,k))/(obj.Ndft1*obj.Ndft2);
	    end;
	    x = x(:);
        end
        
    end
end
