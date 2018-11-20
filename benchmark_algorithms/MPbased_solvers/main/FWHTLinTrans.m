classdef FWHTLinTrans < LinTrans
    % FWHTLinTrans:  Linear transform class for the fast Walsh-Hadamard
    % Transform
    
    properties
        %Signal model is y = Ax. y and x are vectorizations of a signal
        %which is 1-dimensional
        
        %vector giving the size of the signal x. The signal x is
        %assumed to be the complete output of a FWHT with no samples
        %omitted. Each entry of xSize should be an integer multiple of the
        %corresponding entry of ySize
        xSize;
        
        %vector specifying the m samples of the ySize signal
        %y. These are indices into the 1-dimensional FWHT output with
        %size ySize. These indices can be converted to 1-length subscripts
        %for addressing the array using the ind2sub function.If set to an
        %empty matrix, y is assumed to be the complete ySize FWHT
        %result vectorized.
        ySamples;
        
        %The logarithm of xSize needed for scaling.
        log_xSize;
        
        %actual vector for sign flip
        signVec
        
        
    end
    
    methods
        
        % Constructor
        function obj = FWHTLinTrans(xSize,ySamples,randSign)
            
            %Default to A returning the complete DCT
            if nargin < 2
                ySamples = 1:2^(ceil(log2(xSize)));
            end
            
            %Check for integer multiples
            if mod(log2(xSize),1) ~= 0
                warning('The length of the signal is not a power of 2. The signal will be zero padded to the next highest power of 2.')
            end
            
            %Force ySamples to be a column vector
            ySamples = ySamples(:);
            
            %Set the properties
            obj = obj@LinTrans;
            obj.ySamples = ySamples;
            obj.log_xSize = ceil(log2(xSize));
            obj.xSize = 2^obj.log_xSize;
            
            if nargin > 3 && randSign
                obj.signVec = sign(rand(xSize,1)-0.5);
            else
                obj.signVec = ones(xSize,1);
            end
            
        end
        
        % Size
        function [m,n] = size(obj)
            
            m = length(obj.ySamples);
            
            %Length of signal x
            n = obj.xSize;
            
            if m>n
                error('The number of measurements cannot be greater than the number of signal elements.')
            end
            
        end
        
        % Replace ySamples with m sample locations chosen uniformly at
        % random
        function ind = ySamplesRandom(obj,m)
            
            %Determine largest possible sample index
            indMax = obj.xSize;
            
            %Generate a random ordering of all possible ySamples indices
            indSamples = randperm(indMax);
            
            %keep only the first m of these random sample indices and
            %ensure that it is a column vector
            ind = reshape(indSamples(1:m),[],1);
        end
        
        %set ySamples using an mx1 matrix of subscripts, rather than
        %providing the indices. This may be more natural for some users.
        function ySamplesSetFromSubScripts(obj,subVals)
            
            %Check size
            if size(subVals,2) ~= 1
                error('Input should be Mx1.')
            end
            
            obj.ySamples = subVals(:);
            
        end
        
        
        % Matrix multiply
        function y = mult(obj,x)
            
            %find number of signal columns
            t = size(x,2);
            
            if t ==1
            
                % The mex function for fastWHtrans does not work on complex
                % data.  Must handle this cases manually
                if any(~isreal(x(:)))
                    realx = real(x); imagx = imag(x);

                    realy = fastWHtrans(obj.signVec.*realx);
                    imagy = fastWHtrans(obj.signVec.*imagx);

                    %combine real and imaginary parts
                    y = realy + 1i*imagy;

                else
                    %Compute the actual FWHT.
                    y = fastWHtrans(obj.signVec.*x);  

                end  
                
            %Use the fastWHtrans for MMV case, however since its not built
            %for the MMV problem, only loop if number of columns is
            %sufficiently small.
            elseif t>1 && t<=max(100*obj.log_xSize,1)
                
                %preallocate memory
                y = zeros(obj.xSize,t);
                for i = 1:t
                    
                    % The mex function for fastWHtrans does not work on complex
                    % data.  Must handle this cases manually
                    if any(~isreal(x(:,i)))
                        realx = real(x(:,i)); imagx = imag(x(:,i));

                        realy = fastWHtrans(obj.signVec.*realx);
                        imagy = fastWHtrans(obj.signVec.*imagx);

                        %combine real and imaginary parts
                        y(:,i) = realy + 1i*imagy;

                    else
                        
                        %Compute the actual FWHT.
                        y(:,i) = fastWHtrans(obj.signVec.*x(:,i));  
                        
                    end 
                                    
                end
                
            else
                y = fwht(obj.signVec.*x)*2^(obj.log_xSize/2);
                
            end
                      
            %Downsample
            if ~isempty(obj.ySamples)
                y = y(obj.ySamples,:);
            end
            
        end
        
        
        % Matrix multiply transpose
        function x = multTr(obj,y)
            
            %find number of signal columns
            t = size(y,2);
            
            %First, need to expand the input with zeros if the A output is
            %subsampled
            if ~isempty(obj.ySamples)

                %Preallocate the full length vector
                yFull = zeros(obj.xSize,t);

                %Assign the input values
                yFull(obj.ySamples,:) = y;

            else %Otherwise just use y

                yFull = y;

            end
            
            %perform multTr for single vector case
            if t ==1

                % The mex function for fastWHtrans does not work on complex
                % data.  Must handle this cases manually
                if any(~isreal(y(:)))
                    realy = real(yFull); imagy = imag(yFull);

                    realx = obj.signVec.*fastWHtrans(realy);
                    imagx = obj.signVec.*fastWHtrans(imagy);

                    %combine real and imaginary parts
                    x = realx + 1i*imagx;

                else
                    %Compute the actual FWHT. By our assumptions, we always want
                    %to compute an FWHT of xSize.
                    x = obj.signVec.*fastWHtrans(yFull);
                end
                
            %Use the fastWHtrans for MMV case, however since its not built
            %for the MMV problem, only loop if number of columns is
            %sufficiently small.
            elseif t>1 && t<=max(100*obj.log_xSize,1)
                
                %preallocate memory
                x = zeros(obj.xSize,t);
                
                for i = 1:t
                    
                    % The mex function for fastWHtrans does not work on complex
                    % data.  Must handle this cases manually
                    if any(~isreal(yFull(:,i)))
                        realy = real(yFull(:,i)); imagy = imag(yFull(:,i));

                        realx = obj.signVec.*fastWHtrans(realy);
                        imagx = obj.signVec.*fastWHtrans(imagy);

                        %combine real and imaginary parts
                        x(:,i) = realx + 1i*imagx;

                    else
                        
                        %Compute the actual FWHT.
                        x(:,i) = obj.signVec.*fastWHtrans(yFull(:,i));  
                        
                    end 
                    
                end
               
            %Looping takes too long for MMV case, use MATLAB's, built in
            %function
            else   
                x = obj.signVec.*ifwht(yFull)*2^(-obj.log_xSize/2);
            end
            
        end
        
        
        % Matrix multiply with square
        function y = multSq(obj,x)
            
            %get number of measurements
            [m,~] = obj.size();
            
            %find number of signal columns
            t = size(x,2);
            
            %All outputs are equal to the scaled sum
            y = ones(m,t)*(sum(x)*2^(-obj.log_xSize));
                      
        end
        
        
        % Matrix multiply transpose
        function x = multSqTr(obj,y)
            
            %get number of measurements
            [~,n] = obj.size();
            
            %find number of signal columns
            t = size(y,2);
            
            %All outputs are equal to the scaled sum
            x = ones(n,t)*(sum(y)*2^(-obj.log_xSize));
                    
        end
        
        
    end
end
