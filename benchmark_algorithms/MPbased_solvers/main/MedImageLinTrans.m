% CLASS: MedImageLinTrans
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: LinTrans
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The MedImageLinTrans operator is used to describe a medical imaging
%   data acquisition process in which undersampled, noise-free k-space data
%   of an image is acquired.  The image is assumed to be sparse in the
%   wavelet domain (thus, the unknown signal, X, being recovered is the
%   wavelet decomposition of the image).  The linear transformation Z = A*X
%   can be expressed as the composition of three operations,
%                           Z = M*F*W'*X,
%   where W' is an inverse wavelet transform (i.e., synthesis operator), F
%   is a 2-D discrete Fourier transform, and M is a binary subsampling mask
%   that selects only a subset of the available k-space data.
%
% PROPERTIES (State variables)
%   MASK      	An NY-by-NX dimensional binary subsampling mask, where NY
%               is the number of rows in the complete k-space plane (the ky
%               component), and NX is the number of columns in the complete
%               k-space plane (the kx component)
%   wname       The character string identifier for the desired type of
%               wavelet filter, e.g., 'db4' for a Daubechies-4 wavelet
%               filter.  (Type "help wfilters" for a list of available
%               filters)
%   L           The number of wavelet decomposition levels
%   S           MATLAB's "bookkeeping matrix" associated with the desired
%               L-level wname wavelet decomposition.  This matrix can be
%               obtained as the second output of the command 
%               "wavedec2(I, L, wname)", where I is a matrix whose
%               dimensions match the dimensions of the image (NY-by-NX)
%
% METHODS (Subroutines/functions)
%   MedImageLinTrans(MASK, wname, L, S)
%       - Default constructor.
%   [M, N] = size(obj)
%       - Returns the size of the implicit A operator
%   z = mult(obj, x)
%       - Perform the forward matrix multiplication
%   x = multTr(obj, z)
%       - Perform the adjoint matrix multiplication
%   z = multSq(obj, x)
%       - Approximate the operation (A.^2)*x
%   x = multSqTr(obj, z)
%       - Approximate the operation (A.^2)'*x
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 05/21/12
% Change summary: 
%       - Created (05/21/12; JAZ)
% Version 0.1
%

classdef MedImageLinTrans < LinTrans
    
    properties (SetAccess = immutable)
        MASK; 	% NY-by-NX dimensional binary k-space subsampling mask
        wname;  % Character string identifier for desired wavelet filter
        L;      % Number of levels of the wavelet decomposition
        S;      % MATLAB's wavedec2 "bookkeeping matrix"
    end
    
    properties (Hidden)
        NX;     % Number of columns in k-space (kx)
        NY;     % Number of rows in k-space (ky)
        Nz;     % Length of the z vector (# of acquired k-space points)
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = MedImageLinTrans(MASK, wname, L, S)
            obj = obj@LinTrans;
            if nargin ~= 4
                error('Constructor must have four input arguments')
            else
                [obj.NY, obj.NX] = size(MASK);
                if ~all(MASK(:) == 0 | MASK(:) == 1)
                    error('Entries of M must be either 0 or 1')
                end
                obj.Nz = sum(MASK(:) == 1);
                obj.MASK = MASK;
                if ~isa(wname, 'char')
                    error('wname must be a character string identifier')
                end
                obj.wname = wname;
                if L <= 0
                    error('L must be a strictly positive integer')
                end
                obj.L = L;
                if size(S, 1) ~= (L + 2) || size(S, 2) ~= 2
                    error('S does not appear to be a valid bookkeeping matrix')
                elseif S(L+2,1) ~= obj.NY || S(L+2,2) ~= obj.NX
                    error('Sizing disagreement between MASK and S')
                end
                obj.S = S;
            end
        end
        
        
        % *****************************************************************
        %                           SIZE METHOD
        % *****************************************************************
        function [M, N] = size(obj)
            % Return the size of the vectorized output Z in M, and the size
            % of the vectorized wavelet coefficients X in N
            N = sum(3*prod(obj.S(1:obj.L+1,:), 2)) - 2*prod(obj.S(1,:));
            M = obj.Nz;                     % # of k-space samples
        end
        
        % *****************************************************************
        %                  MATRIX MULTIPLICATION METHOD
        % *****************************************************************
        function z = mult(obj, x)
            
            % First synthesize image from wavelet coefficients
            IMG = waverec2(x.', obj.S, obj.wname);
            
            % Check for any sizing problems
            if size(IMG, 1) ~= obj.NY || size(IMG, 2) ~= obj.NX
                error('Wavelet reconstruction did not yield correct size')
            end

            % Next compute complete k-space data using FFT
            KSPC = (1/sqrt(obj.NY)) * (1/sqrt(obj.NX)) * fft2(IMG);

            % Finally, keep only those points in k-space for which MASK is
            % equal to 1
            z = KSPC(obj.MASK == 1);
        end

        % *****************************************************************
        %                  ADJOINT MULTIPLICATION METHOD
        % *****************************************************************
        function x = multTr(obj, z)

            % First, upsample from the decimated k-space domain to the
            % complete k-space domain by filling in zeros at unacquired
            % points
            KSPC = zeros(obj.NY, obj.NX);
            KSPC(obj.MASK == 1) = z;
            
            % Next, transform to the image domain using a 2-D IFFT
            IMG = sqrt(obj.NY) * sqrt(obj.NX) * ifft2(KSPC);
            
            % Finally, compute the L-level wname wavelet decomposition
            x = wavedec2(IMG, obj.L, obj.wname).';
        end

        % *****************************************************************
        %                  SQUARED MULTIPLICATION METHOD
        % *****************************************************************
        function z = multSq(obj, x)
            
            % Get number of measurements
            [M, ~] = obj.size();
            
            % All outputs are approximated by a scaled sum
            z = kron(ones(M,1),sum(x)/M);
        end
        
        
        % *****************************************************************
        %        	  SQUARED TRANSPOSE MULTIPLICATION METHOD
        % *****************************************************************
        function x = multSqTr(obj, z)
            
            % Get number of measurements
            [M, N] = obj.size();
            
            % All outputs are approximated by a scaled sum
            x = kron(ones(N,1),sum(z)/M);
        end
        
    end
end