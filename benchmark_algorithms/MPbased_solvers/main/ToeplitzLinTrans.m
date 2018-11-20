classdef ToeplitzLinTrans < LinTrans
    % ToeplitzLinTrans: Implements a forward operator that is a (possibly
    % decimated) Toeplitz matrix.
    
    %Deny user direct access to these, as the methods rely on them being
    %set correctly. May want to add methods to change them individually at
    %a later time. 
    properties (SetAccess = private)
        
        %Defining first row/column of Toeplitz matrix
        row;
        col;
        
        %Specify which rows/columns are included in the matrix. These are
        %1:Mfull and 1:Nfull to yield the full matrix
        whichR;
        whichC;
        
        %Various size variables
        M; N; %Overall operator size
        Mfull; Nfull; %Size of underlying Toeplitz matrix
        Nfft; %size of required FFT
        
        %Impulse response and impulse response squared, and reverses
        imp;
        imp2;
        imp_rev;
        imp_rev2;
        
    end
    
    methods
        
        % Constructor. Pass integers whichR/whichC to decimate, or may pass
        % vectors of actual values. The vectors contain integer locations
        % of rows/columns to be INCLUDED.
        function obj = ToeplitzLinTrans(row,col,whichR,whichC)
            
            %Super constructor
            obj = obj@LinTrans;
            
            %Assign row and col
            obj.row = reshape(row,1,[]);
            obj.col = reshape(col,[],1);
            
            %Define size of complete Toeplitz matrix
            obj.Nfull = numel(row);
            obj.Mfull = numel(col);
            
            %Check for more arguments
            if nargin > 2
                
                %Check for decimation vs. indices
                if isscalar(whichR)
                    obj.whichR = 1:whichR:obj.Mfull;
                else
                    obj.whichR = whichR(:);
                end
                
                %Check for decimation vs. indices
                if isscalar(whichC)
                    obj.whichC = 1:whichC:obj.Nfull;
                else
                    obj.whichC = whichC(:);
                end
            else
                
                %Just set to full length
                obj.whichR = 1:obj.Mfull;
                obj.whichC = 1:obj.Nfull;
            end
            
            %Ensure columns
            obj.whichR = obj.whichR(:);
            obj.whichC = obj.whichC(:);
            
            %Define effective size variables
            obj.N = numel(obj.whichC);
            obj.M = numel(obj.whichR);
            
            %Determine FFT size
            obj.Nfft = 2^ceil(log2(obj.Mfull+obj.Nfull-1));
            
            %Impulse response and impulse response squared
            obj.imp = [flipud(obj.row(:));obj.col(2:end)];
            obj.imp2 = abs(obj.imp).^2;
            
            %Reversed impule responses
            obj.imp_rev = conj(flipud(obj.imp));
            obj.imp_rev2 = abs(obj.imp_rev).^2;
            
            
        end
        
        % Size
        function [m,n] = size(obj)
            
            %Get sizes
            m = obj.M;
            n = obj.N;
            
        end
        
        % Matrix multiply
        function y = mult(obj,x)
            
            %Construct input vector
            xF = zeros(obj.Nfull,1);
            if numel(x) ~= obj.N
                error('Length of input vector is incorrect')
            else
                xF(obj.whichC) = x;
            end
            
            %Carry out the fast convolution
            yF = ifft(fft(obj.imp,obj.Nfft).*fft(xF,obj.Nfft));
            
            %Trim the result
            yF = yF(obj.Nfull:end);
            
            %Decimate the output
            y = yF(obj.whichR);
            
            
        end
        
        
        % Matrix multiply transpose
        function x = multTr(obj,y)
            
            %Construct input vector
            yF = zeros(obj.Mfull,1);
            if numel(y) ~= obj.M
                error('Length of input vector is incorrect')
            else
                yF(obj.whichR) = y;
            end
            
            %Compute the convolution
            xF = ifft(fft(obj.imp_rev,obj.Nfft).* ...
                fft(yF,obj.Nfft));
            
            %Trim the result
            xF = xF(obj.Mfull:end);
            
            %Decimate the output
            x = xF(obj.whichC);
            
            
        end
        
        
        
        
        % Matrix multiply with square
        function y = multSq(obj,x)
            
            %Construct input vector
            xF = zeros(obj.Nfull,1);
            if numel(x) ~= obj.N
                error('Length of input vector is incorrect')
            else
                xF(obj.whichC) = x;
            end
            
            %Carry out the fast convolution
            yF = ifft(fft(obj.imp2,obj.Nfft).*fft(xF,obj.Nfft));
            
            %Trim the result
            yF = yF(obj.Nfull:end);
            
            %Decimate the output
            y = yF(obj.whichR);
            
        end
        
        
        % Matrix multiply transpose squared
        function x = multSqTr(obj,y)
            
            %Construct input vector
            yF = zeros(obj.Mfull,1);
            if numel(y) ~= obj.M
                error('Length of input vector is incorrect')
            else
                yF(obj.whichR) = y;
            end
            
            %Compute the convolution
            xF = ifft(fft(obj.imp_rev2,obj.Nfft).* ...
                fft(yF,obj.Nfft));
            
            %Trim the result
            xF = xF(obj.Mfull:end);
            
            %Decimate the output
            x = xF(obj.whichC);
            
        end
        
        
    end
end