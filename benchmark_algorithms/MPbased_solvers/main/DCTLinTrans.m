classdef DCTLinTrans < LinTrans
    % DCTLinTrans:  Linear transform class for discrete cosine transform.
    % Currently supports only 1 dimensional dcts.
    
    properties
        %Signal model is y = Ax. y and x are vectorizations of a signal
        %which is 1-dimensional
        
        %vector giving the size of the DCT/IDCT used to compute y.
        %y may represent a further subsampling of this grid, for example
        %when randomized Fourier measurements are collected.
        ySize;
        
        %vector giving the size of the signal x. The signal x is
        %assumed to be the complete output of a DCT/IDCT with no samples
        %omitted. Each entry of xSize should be an integer multiple of the
        %corresponding entry of ySize
        xSize;
        
        %vector specifying the m samples of the ySize signal
        %y. These are indices into the 1-dimensional DCT/IDCT output with
        %size ySize. These indices can be converted to 1-length subscripts
        %for addressing the array using the ind2sub function.If set to an
        %empty matrix, y is assumed to be the complete ySize DCT/IDCT
        %result vectorized.
        ySamples;
        
        %Determines whether A is a DCT (true) or IDCT (false). Naturally,
        %A Hermetian is the opposite operator.
        domain;
        
    end
    
    methods
        
        % Constructor
        function obj = DCTLinTrans(ySize,xSize,domain,ySamples)
            
            %Default to A returning the complete DCT
            if nargin < 4
                ySamples = [];
            end                
            if nargin < 3
                domain = true;
            end
            if nargin < 2
                xSize = ySize;
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
        function [m,n] = size(obj)
            
            %Number of measurements
            if isempty(obj.ySamples)
                m = prod(obj.ySize);
            else
                m = length(obj.ySamples);
            end
            
            %Length of signal x
            n = prod(obj.xSize);
            
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
            if obj.domain %A is a DCT
                
                
                %Compute the actual DCT. By our assumptions, we always want
                %to compute an DCT of xSize. Note that x is already xSize
                %here, so the size argument is actually redundant
                result = dct(x,obj.xSize);
                
            else %A is an IDCT
                
                %Compute the actual DCT. By our assumptions, we always want
                %to compute an DCT of xSize. Note that x is already xSize
                %here, so the size argument is actually redundant
                result = idct(x,obj.xSize);
                
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
            if obj.domain %A is a DCT
                y = result;
            else
                y = result*prod(obj.xSize);
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
            if ~obj.domain %A Hermetian is a DCT
                
                
                %Compute the actual DCT. By our assumptions, we always want
                %to compute an DCT of xSize.
                result = dct(yFull,obj.xSize);
                
            else %A Hermetian is an IDCT
                
                %Compute the actual DCT. By our assumptions, we always want
                %to compute an DCT of xSize.
                result = idct(yFull,obj.xSize);
                
            end
            
            
            %Handle Q = 1 differently
            if Q > 1
                %Now we have to scale the result to enforce unit-norm columns
                %in the A operator
                if ~obj.domain %A Hermetian is a DCT
                    x = result(:);
                else
                    x = result(:);
                end
            else
                
                %Now we have to scale the result to enforce unit-norm columns
                %in the A operator
                if ~obj.domain %A Hermetian is a DCT
                    x = result;
                else
                    x = result;
                end
            end
            
        end
        
        
        % Matrix multiply with square
        function y = multSq(obj,x)
            
            %get number of measurements
            [m,n] = obj.size();
            
            %All outputs are equal to the scaled sum
            y = ones(m,1)*(sum(x)/n);
            
            
            
        end
        
        
        % Matrix multiply transpose
        function x = multSqTr(obj,y)
            
            %get number of measurements
            [~,n] = obj.size();
            
            %All outputs are equal to the scaled sum
            x = ones(n,1)*(sum(y)/n);
            
            
        end
        
        
    end
end
