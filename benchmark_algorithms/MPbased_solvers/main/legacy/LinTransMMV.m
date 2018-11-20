classdef LinTransMMV < LinTrans
    % LinTransMMVHelper:  This class lets you use any LinTrans object for
    % Multiple Measurement Vector problems.
    
    properties
        
        %Only class member is a LinTrans object that operators on a vector
        %signal.
        A;
        
    end
    
    methods
        
        % Constructor
        function obj = LinTransMMV(A)
            
            %Assign the multiply
            obj.A = A;
            
            
        end
        
        % Size
        function [m,n] = size(obj)
            
            %Use the vector multiply object to get size
            [m,n] = obj.A.size();
            
        end
        
        % Matrix multiply
        function y = mult(obj,x)
            
            %Determine number of samples
            T = size(x,2);
            [m,~] = obj.size;
            
            %Preallocate the result
            y = zeros(m,T);
            
            %Carry out multiplies
            for kk = 1:T
                y(:,kk) = obj.A.mult(x(:,kk));
            end
            
        end
        
        
        % Matrix multiply transpose
        function x = multTr(obj,y)
            
            %Determine number of samples
            T = size(y,2);
            [~,n] = obj.size;
            
            %Preallocate the result
            x = zeros(n,T);
            
            %Carry out multiplies
            for kk = 1:T
                x(:,kk) = obj.A.multTr(y(:,kk));
            end
            
            
        end
        
        
        % Matrix multiply with square
        function y = multSq(obj,x)
            
            %Determine number of samples
            T = size(x,2);
            [m,~] = obj.size;
            
            %Preallocate the result
            y = zeros(m,T);
            
            %Carry out multiplies
            for kk = 1:T
                y(:,kk) = obj.A.multSq(x(:,kk));
            end
            
            
            
        end
        
        
        % Matrix multiply transpose
        function x = multSqTr(obj,y)
            
            %Determine number of samples
            T = size(y,2);
            [~,n] = obj.size;
            
            %Preallocate the result
            x = zeros(n,T);
            
            %Carry out multiplies
            for kk = 1:T
                x(:,kk) = obj.A.multSqTr(y(:,kk));
            end
            
            
        end
        
        
    end
end