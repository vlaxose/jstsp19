classdef LTOperator
    % LTOperator:  A wrapper class to perform operations with LinTrans
    % objects.  Note that this is a value class, so the state can be
    % changed without changing the original object
    properties (Access = private)
        % Flag indicating if next mtimes should use a transpose operation
        adjoint = false;    
        
        % Flag indicating if next mtimes should use squared operation
        absSqSet = false;     
        
        % Reference to LinTrans object
        lt;
    end
        
    methods 
        % Constructor
        function obj = LTOperator(lt, adjoint, absSqSet)
            obj.lt          = lt;
            obj.adjoint     = adjoint;
            obj.absSqSet    = absSqSet;            
        end
        
        % Multiplication operator
        % 
        % z = A*x           => z = A.mult(x)
        % z = A'*x          => z = A.multTr(x)
        % z = A.absSq()*x   => z = A.multSq(x)
        % z = A.absSq()'*x  => z = A.multSqTr(x) 
        function z = mtimes(obj,x)
            if (obj.adjoint) && (obj.absSqSet)
                z = obj.lt.multSqTr(x);
            elseif (obj.adjoint) 
                z = obj.lt.multTr(x);
            elseif (obj.absSqSet) 
                z = obj.lt.multSq(x);                
            else
                z = obj.lt.mult(x);
            end
            
        end
        
        % Conjugate transpose.         
        function obj1 = ctranspose(obj)
            obj1 = obj;     
            obj1.adjoint = ~obj.adjoint;
        end
        
        % Absolute value squared.       
        function obj1 = absSq(obj)
            if (obj.absSqSet) 
                error('The object is already squared');
            end
            obj1 = obj;
            obj1.absSqSet = true;
        end
        
        % Size
        function [m,n] = size(obj)            
            if (obj.adjoint)
                [n,m] = size(obj.lt);
            else
                [m,n] = size(obj.lt);
            end                
        end
        
        % Matrix multiply:  z = A*x
        function z = mult(obj,x)
            z = obj*x;
        end        

        % Matrix multiply transpose:  x = A'*z
        function x = multTr(obj,z)            
            x = obj'*z;        
        end

        % Matrix multiply with square:  z = (abs(A).^2)*x
        function z = multSq(obj,x)
            z = obj.absSq()*x;
        end

        % Matrix multiply with componentwise square transpose:  
        % x = (abs(A).^2)'*z
        function x = multSqTr(obj,z)
            x = obj.absSq()'*z;
        end
    end
        
end