classdef FourierLinTrans < LinTrans
    % FourierTrans:  Linear transform class for arbitrary dimensional
    % DFTs. This implementation draws heavily on Fourier operator code
    % provided by Christian Austin at The Ohio State University.
    
    properties
        %Signal model is y = Ax. y and x are vectorizations of a signal
        %which is Q-dimensional
        
        %1xQ vector giving the size of the DFT/IDFT used to compute y.
        %y may represent a further subsampling of this grid, for example
        %when randomized Fourier measurements are collected.
        ySize;
        
        %1xQ vector giving the size of the signal x. The signal x is
        %assumed to be the complete output of a DFT/IDFT with no samples
        %omitted. Each entry of xSize should be an integer multiple of the
        %corresponding entry of ySize
        xSize;
        
        %mx1 vector specifying the m samples of the ySize signal
        %y. These are indices into the Q-dimensional DFT/IDFT output with
        %size ySize. These indices can be converted to Q-length subscripts
        %for addressing the array using the ind2sub function.If set to an
        %empty matrix, y is assumed to be the complete ySize DFT/IDFT
        %result vectorized.
        ySamples;
        
        %Determines whether A is a DFT (true) or IDFT (false). Naturally,
        %A Hermetian is the opposite operator.
        domain;
        
    end
    
    methods
        
        % Constructor
        function obj = FourierLinTrans(ySize,xSize,domain,ySamples)
            
            %Default to A returning the complete DFT
            if nargin < 4
                ySamples = [];
                domain = true;
            elseif nargin < 3
                domain = true;
            end
            
            %Check for integer multiples
            if sum(mod(xSize,ySize)) ~= 0
                error('xSize should be integer multiples of ySize')
            end
            
            %Force ySamples to be a column vector
            ySamples = ySamples(:);
            
            %Set the properties
            obj = obj@LinTrans;
            obj.xSize = xSize;
            obj.ySize = ySize;
            obj.ySamples = ySamples;
            obj.domain = domain;
            
        end
        
        % Size
        function [m,n] = size(obj,dim)
            
            %Number of measurements
            if isempty(obj.ySamples)
                m = prod(obj.ySize);
            else
                m = length(obj.ySamples);
            end
            
            %Length of signal x
            n = prod(obj.xSize);
            
            %Set dimensions
            obj.dims = [m n];
            
            if nargin>1 % a specific dimension was requested
                if dim>2
                    m=1;
                else
                    m=obj.dims(dim);
                end
            elseif nargout<2  % all dims in one output vector
                m=obj.dims;
            else % individual outputs for the dimensions
                m=obj.dims(1);
                n=obj.dims(2);
            end
        end
        
        % Replace ySamples with m sample locations chosen uniformly at
        % random
        function ySamplesRandom(obj,m)
            
            %Determine largest possible sample index
            indMax = prod(obj.ySize);
            
            %Generate a random ordering of all possible ySamples indices
            indSamples = randperm(indMax);
            
            %keep only the first m of these random sample indices and
            %ensure that it is a column vector
            obj.ySamples = reshape(indSamples(1:m),[],1);
        end
        
        % Replace ySamples with m sample locations chosen in a block 
        % starting at index mstart>=1
        function ySamplesBlock(obj,m,mstart)
            
            %Determine largest possible sample index
            indMax = prod(obj.ySize);
	    if m+mstart-1 > indMax
	      error('m+mstart-1 must be <= prod(obj.ySize)')
	    end;
            
            %extract sample indices into column vector
            obj.ySamples = reshape(mstart:mstart+m-1,[],1);
        end
        
        %return an mxQ matrix of the subscripts associated with ySamples
        function result = ySamplesSubscripts(obj)
            
            %Ensure that ySamples is a column vector
            obj.ySamples = obj.ySamples(:);
            
            %Determine dimensionality
            Q = length(obj.ySize);
            
            %Preallocate a cell array to hold the sample indices
            result = cell(1,Q);
            
            %Compute indices into the Q-dimensional array
            [result{:}] = ind2sub(obj.ySize,obj.ySamples);
            
            %Store the result as a matrix
            result = cell2mat(result);
            
        end
        
        %set ySamples using an mxQ matrix of subscripts, rather than
        %providing the indices. This may be more natural for some users.
        function ySamplesSetFromSubScripts(obj,subVals)
            
            %Determine dimensionality
            Q = length(obj.ySize);
            
            %Check size
            if size(subVals,2) ~= Q
                error('Input should be mxQ.')
            end
            
            if Q > 1
                %Convert to cell array
                subValsCell = mat2cell(subVals,size(subVals,1),ones(Q,1));
                
                %Save the result
                obj.ySamples = sub2ind(obj.ySize,subValsCell{:});
                
            else
                obj.ySamples = subVals(:);
            end
            
        end
        
        
        % Matrix multiply
        function y = mult(obj,x)
            
            %Determine dimensinality
            Q = length(obj.xSize);
            
            %get number of measurements
            [m,~] = obj.size();
            
            %Reshape x if Q > 1, i.e. not a 1D signal
            if Q > 1
                x = reshape(x,obj.xSize);
            end
            
            %Check which domain x is in
            if obj.domain %A is a DFT
                
                
                %Compute the actual DFT. By our assumptions, we always want
                %to compute an FFT of xSize. Note that x is already xSize
                %here, so the size argument is actually redundant
                if Q == 1
                    result = fft(x,obj.xSize);
                else
                    result = fftn(x,obj.xSize);
                end
                
            else %A is an IDFT
                
                %Compute the actual DFT. By our assumptions, we always want
                %to compute an FFT of xSize. Note that x is already xSize
                %here, so the size argument is actually redundant
                if Q == 1
                    result = ifft(x,obj.xSize);
                else
                    result = ifftn(x,obj.xSize);
                end
                
            end
            
            %Handle this differently for Q=1 to support MMV
            if Q > 1
                %If ySize is smaller than xSize, then we need to trim the
                %resulting outputs. One can view this step as bandlimiting due
                %to the implicit downsampling in the transform domain.
                trimInd = cell(1,Q);
                for kk=1:Q
                    trimInd{kk} = 1:obj.ySize(kk);
                end
                result = result(trimInd{:});
                
                %Vecotrize the result
                result = result(:);
                
                %If only a subset of samples are returned, we have to
                %downselect
                if ~isempty(obj.ySamples)
                    result = result(obj.ySamples);
                end
            else
                
                %Trim
                result = result(1:obj.ySize,:);
                
                %Downsample
                if ~isempty(obj.ySamples)
                    result = result(obj.ySamples,:);
                end
            end
            
            
            
            %Now we have to scale the result to enforce unit-norm columns
            %in the A operator
            if obj.domain %A is a DFT
                y = result /sqrt(m);
            else
                y = result/sqrt(m)*prod(obj.xSize);
            end
            
            
            
        end
        
        
        % Matrix multiply transpose
        function x = multTr(obj,y)
            
            %Determine dimensinality
            Q = length(obj.xSize);
            
            %get number of measurements
            [m,~] = obj.size();
            
            %First, need to expand the input with zeros if the A output is
            %subsampled
            if ~isempty(obj.ySamples)
                
                %Preallocate the full length vector
                yFull = zeros(prod(obj.ySize),size(y,2));
                
                %Assign the input values
                yFull(obj.ySamples,:) = y;
                
            else %Otherwise just use y
                
                yFull = y;
                
            end
            
            %Reshape the input if y is not 1D
            if Q > 1
                yFull = reshape(yFull,obj.ySize);
            end
            
            %Check which domain y is in
            if ~obj.domain %A Hermetian is a DFT
                
                
                %Compute the actual DFT. By our assumptions, we always want
                %to compute an FFT of xSize.
                if Q == 1
                    result = fft(yFull,obj.xSize);
                else
                    result = fftn(yFull,obj.xSize);
                end
                
            else %A Hermetian is an IDFT
                
                %Compute the actual DFT. By our assumptions, we always want
                %to compute an FFT of xSize.
                if Q == 1
                    result = ifft(yFull,obj.xSize);
                else
                    result = ifftn(yFull,obj.xSize);
                end
                
            end
            
            
            %Handle Q = 1 differently
            if Q > 1
                %Now we have to scale the result to enforce unit-norm columns
                %in the A operator
                if ~obj.domain %A Hermetian is a DFT
                    x = result(:) /sqrt(m);
                else
                    x = result(:)/sqrt(m)*prod(obj.xSize);
                end
            else
                
                %Now we have to scale the result to enforce unit-norm columns
                %in the A operator
                if ~obj.domain %A Hermetian is a DFT
                    x = result /sqrt(m);
                else
                    x = result/sqrt(m)*prod(obj.xSize);
                end
            end
            
        end
        
        
        % Matrix multiply with square
        function y = multSq(obj,x)
            
            %get number of measurements
            [m,~] = obj.size();
            
            %All outputs are equal to the scaled sum
            y = kron(ones(m,1),sum(x)/m);
            
            
            
        end
        
        
        % Matrix multiply transpose
        function x = multSqTr(obj,y)
            
            %get number of measurements
            [m,n] = obj.size();
            
            %All outputs are equal to the scaled sum
            x = kron(ones(n,1),sum(y)/m);
            
            
        end
        
        
    end
end
