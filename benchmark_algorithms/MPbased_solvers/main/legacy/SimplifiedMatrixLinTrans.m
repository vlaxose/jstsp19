classdef SimplifiedMatrixLinTrans < LinTrans
    % SimplifiedLinTrans:  Linear transform class with a matrix. The
    % simplified version replace the multSq and multSqTr with simple sum
    % operations. Used to obtain a simplified version of GAMP.
    
    properties
        A;      % Matrix A
        %Asq;    % Asq = (A.^2)
    end
    
    methods
        
        % Constructor
        function obj = SimplifiedMatrixLinTrans(A)
            obj = obj@LinTrans;
            obj.A = A;
            %obj.Asq = (abs(A).^2);
        end
        
        % Size
        function [m,n] = size(obj)
            [m,n] = size(obj.A);
        end
        
        % Matrix multiply
        function y = mult(obj,x)
            y = obj.A*x;
        end
        % Matrix multiply transpose
        function y = multTr(obj,x)
            y = obj.A'*x;
        end
        
        
        % Matrix multiply with square
        function y = multSq(obj,x)
            
            %get number of measurements
            [m,~] = obj.size();
            
            %All outputs are equal to the scaled sum
            y = ones(m,1)*sum(x)/m;
            
        end
        
        
        % Matrix multiply transpose
        function x = multSqTr(obj,y)
            
            %get number of measurements
            [m,n] = obj.size();
            
            %All outputs are equal to the scaled sum
            x = ones(n,1)*sum(y)/m;
            
            
        end
        
        
    end
end